import os
import glob
import json
import subprocess
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------- CONFIGURATION --------------------
# File to store recent repo URLs
RECENT_REPOS_FILE = "recent_repos.json"

# Local directory where repos will be cloned
CLONE_BASE_DIR = os.path.join(os.getcwd(), "cloned_repos")
if not os.path.exists(CLONE_BASE_DIR):
    os.makedirs(CLONE_BASE_DIR)

# Embedding API details
EMBED_API_URL = "http://localhost:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"

# Code file extensions to process
FILE_EXTENSIONS = [".py", ".js", ".java", ".cpp", ".c", ".ts", ".go", ".rb", ".php"]

# Chunking parameter (character-based for now)
CHUNK_SIZE = 512

# -------------------- HELPER FUNCTIONS --------------------
def load_recent_repos():
    if os.path.exists(RECENT_REPOS_FILE):
        try:
            with open(RECENT_REPOS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[DEBUG] Failed to load recent repos: {e}")
    return []

def save_recent_repos(repo_list):
    try:
        with open(RECENT_REPOS_FILE, "w", encoding="utf-8") as f:
            json.dump(repo_list, f)
    except Exception as e:
        print(f"[DEBUG] Failed to save recent repos: {e}")

def add_repo_to_recent(repo_url):
    repos = load_recent_repos()
    if repo_url in repos:
        repos.remove(repo_url)
    repos.insert(0, repo_url)
    # Keep only the last 10 entries
    repos = repos[:10]
    save_recent_repos(repos)
    return repos

def clone_or_update_repo(repo_url):
    """
    Clones the given GitHub repo URL into CLONE_BASE_DIR.
    If the repo was already cloned, it pulls the latest changes.
    Returns the local path of the repo.
    """
    # Extract a simple repo name from the URL
    repo_name = repo_url.rstrip("/").split("/")[-1]
    local_path = os.path.join(CLONE_BASE_DIR, repo_name)
    
    if os.path.exists(local_path):
        # Update the repo
        try:
            subprocess.check_call(["git", "-C", local_path, "pull"])
        except Exception as e:
            messagebox.showerror("Git Error", f"Failed to update repo: {e}")
    else:
        # Clone the repo
        try:
            subprocess.check_call(["git", "clone", repo_url, local_path])
        except Exception as e:
            messagebox.showerror("Git Error", f"Failed to clone repo: {e}")
            return None
    return local_path

def list_code_files(repo_path, extensions):
    """Recursively list all code files in the repository matching the given extensions."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(repo_path, f"**/*{ext}"), recursive=True))
    return files

def read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[DEBUG] Error reading {filepath}: {e}")
        return ""

def chunk_text(text, max_length=CHUNK_SIZE):
    """Split text into fixed-size chunks."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def embed_chunk(chunk, model=EMBED_MODEL):
    """Call the embedding API to embed a given chunk."""
    payload = {"model": model, "input": chunk}
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("embedding", None)
    except Exception as e:
        print(f"[DEBUG] Embedding API error: {e}")
        return None

def process_repo(repo_local_path):
    """
    Processes the repository by traversing code files, chunking them,
    and embedding each chunk. Returns a list of embedding data.
    """
    code_files = list_code_files(repo_local_path, FILE_EXTENSIONS)
    print(f"[DEBUG] Found {len(code_files)} code files.")
    embedding_results = []
    
    for file in code_files:
        content = read_file(file)
        if not content:
            continue
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = embed_chunk(chunk)
            if embedding:
                embedding_results.append({
                    "file": file,
                    "chunk_index": idx,
                    "chunk": chunk,
                    "embedding": embedding
                })
    return embedding_results

# -------------------- GUI FUNCTIONS --------------------
def start_embedding():
    repo_url = repo_url_var.get().strip()
    if not repo_url:
        messagebox.showerror("Input Error", "Please enter or select a repository URL.")
        return

    # Add repo URL to recent list and update dropdown
    repos = add_repo_to_recent(repo_url)
    repo_dropdown["values"] = repos

    # Clone or update the repository
    local_repo = clone_or_update_repo(repo_url)
    if not local_repo:
        return

    # Process the repository: chunk and embed
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Processing repository at {local_repo}...\n")
    output_text.update()

    embeddings = process_repo(local_repo)
    output_text.insert(tk.END, f"\nProcessed {len(embeddings)} chunks.\n")
    
    # Optionally, save the output to a file
    output_file = os.path.join(os.getcwd(), "embedding_output.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, indent=2)
        output_text.insert(tk.END, f"\nEmbeddings saved to: {output_file}\n")
    except Exception as e:
        output_text.insert(tk.END, f"\nError saving output: {e}\n")

def browse_repo():
    # Allow user to choose a local folder instead (if desired)
    folder = filedialog.askdirectory()
    if folder:
        repo_url_var.set(folder)

# -------------------- SETUP GUI --------------------
root = tk.Tk()
root.title("Codebase Embedding GUI")

# Frame for repo URL selection
frame_repo = tk.LabelFrame(root, text="Select GitHub Repository URL")
frame_repo.pack(fill=tk.X, padx=10, pady=5)

repo_url_var = tk.StringVar()
recent_repos = load_recent_repos()
repo_dropdown = ttk.Combobox(frame_repo, textvariable=repo_url_var, width=60)
repo_dropdown["values"] = recent_repos
repo_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

# Button to browse local folder (optional)
browse_btn = tk.Button(frame_repo, text="Browse Folder", command=browse_repo)
browse_btn.pack(side=tk.LEFT, padx=5, pady=5)

# Button to start processing the repository
process_btn = tk.Button(root, text="Process Codebase (Chunk & Embed)", command=start_embedding)
process_btn.pack(pady=10)

# Output text field to display results
output_frame = tk.LabelFrame(root, text="Output")
output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
output_text = tk.Text(output_frame, wrap=tk.WORD, height=20)
output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
