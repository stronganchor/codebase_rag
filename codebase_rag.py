import os
import glob
import json
import subprocess
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import threading

# -------------------- CONFIGURATION --------------------
RECENT_REPOS_FILE = "recent_repos.json"
CLONE_BASE_DIR = os.path.join(os.getcwd(), "cloned_repos")
if not os.path.exists(CLONE_BASE_DIR):
    os.makedirs(CLONE_BASE_DIR)

# Embedding API details (mxbai-embed-large running via Ollama on port 11435)
EMBED_API_URL = "http://localhost:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"

# Code file extensions to process
FILE_EXTENSIONS = [".py", ".js", ".java", ".cpp", ".c", ".ts", ".go", ".rb", ".php"]

# Folders to ignore when processing the repository
SKIP_DIRS = ["getid3", "iso-languages", "plugin-update-checker", "languages", "media", "includes"]

# Global variable to store embeddings (list of dicts)
global_embeddings_data = []

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
    repos = repos[:10]  # Keep last 10 entries
    save_recent_repos(repos)
    return repos

def clone_or_update_repo(repo_url):
    repo_name = repo_url.rstrip("/").split("/")[-1]
    local_path = os.path.join(CLONE_BASE_DIR, repo_name)
    print(f"[DEBUG] Local repo path: {local_path}")
    if os.path.exists(local_path):
        print(f"[DEBUG] Repository exists. Pulling latest changes for {repo_name}...")
        try:
            subprocess.check_call(["git", "-C", local_path, "pull"])
        except Exception as e:
            messagebox.showerror("Git Error", f"Failed to update repo: {e}")
    else:
        print(f"[DEBUG] Cloning repository {repo_url} into {local_path}...")
        try:
            subprocess.check_call(["git", "clone", repo_url, local_path])
        except Exception as e:
            messagebox.showerror("Git Error", f"Failed to clone repo: {e}")
            return None
    return local_path

def list_code_files(repo_path, extensions):
    """Recursively traverse repo_path and return files matching the extensions,
       ignoring directories listed in SKIP_DIRS."""
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        # Filter out directories to skip (case-insensitive)
        dirs[:] = [d for d in dirs if d.lower() not in [skip.lower() for skip in SKIP_DIRS]]
        for f in filenames:
            for ext in extensions:
                if f.endswith(ext):
                    full_path = os.path.join(root, f)
                    files.append(full_path)
                    print(f"[DEBUG] Found file: {full_path}")
                    break
    print(f"[DEBUG] Total code files found: {len(files)}")
    return files

def read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"[DEBUG] Read {len(content)} characters from {filepath}")
        return content
    except Exception as e:
        print(f"[DEBUG] Error reading {filepath}: {e}")
        return ""

def chunk_file_text(text, max_chars):
    """Return the whole text as one chunk if it's under max_chars,
       otherwise split into chunks of size max_chars."""
    if len(text) <= max_chars:
        print(f"[DEBUG] File length {len(text)} <= max_chars {max_chars}, using whole file as one chunk.")
        return [text]
    else:
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        print(f"[DEBUG] File length {len(text)} > max_chars {max_chars}, chunked into {len(chunks)} chunks.")
        return chunks

def embed_chunk(chunk, model=EMBED_MODEL):
    payload = {"model": model, "input": chunk}
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=30)
        print(f"[DEBUG] Embed API response status: {response.status_code}")
        response_text = response.text.strip()
        if response.status_code != 200:
            print(f"[DEBUG] Non-200 response: {response_text}")
            return None
        data = response.json()
        embeddings = data.get("embeddings", None)
        if embeddings is None or not embeddings:
            print(f"[DEBUG] No embeddings returned for chunk. Response: {response_text}")
            return None
        return embeddings[0]
    except Exception as e:
        print(f"[DEBUG] Exception during embedding API call: {e}")
        return None

def process_repo_with_progress(repo_local_path, progress_callback, max_chars):
    """Process the repository: traverse files, read content, chunk, and embed.
       Update progress after each file."""
    code_files = list_code_files(repo_local_path, FILE_EXTENSIONS)
    total_files = len(code_files)
    results = []
    for file_idx, file in enumerate(code_files):
        print(f"[DEBUG] Processing file: {file}")
        content = read_file(file)
        if not content:
            print(f"[DEBUG] File {file} is empty or unreadable.")
            progress_callback(file_idx+1, total_files)
            continue
        # Use whole file as a single chunk unless it exceeds max_chars
        chunks = chunk_file_text(content, max_chars)
        print(f"[DEBUG] File {file} produced {len(chunks)} chunk(s).")
        for idx, chunk in enumerate(chunks):
            preview = chunk.replace('\n', ' ')[:100]
            print(f"[DEBUG] {os.path.basename(file)}: Embedding chunk {idx} (length: {len(chunk)} chars, preview: '{preview}')")
            embedding = embed_chunk(chunk)
            if embedding:
                results.append({
                    "file": file,
                    "chunk_index": idx,
                    "chunk": chunk,
                    "embedding": embedding
                })
                print(f"[DEBUG] {os.path.basename(file)}: Chunk {idx} embedded successfully.")
            else:
                print(f"[DEBUG] {os.path.basename(file)}: Failed to embed chunk {idx}.")
        progress_callback(file_idx+1, total_files)
    print(f"[DEBUG] Total chunks embedded: {len(results)}")
    return results

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def generate_enhanced_prompt(query_text, embeddings_data, top_k=3):
    query_embedding = embed_chunk(query_text)
    if query_embedding is None:
        messagebox.showerror("Error", "Failed to embed the query.")
        return ""
    results = []
    for item in embeddings_data:
        sim = cosine_similarity(query_embedding, item["embedding"])
        results.append((sim, item))
    results.sort(key=lambda x: x[0], reverse=True)
    selected = results[:top_k]
    context_texts = []
    for sim, item in selected:
        context_texts.append(f"File: {item['file']} (Chunk {item['chunk_index']}):\n{item['chunk']}")
        print(f"[DEBUG] Selected chunk (sim={sim:.4f}) from {item['file']} chunk {item['chunk_index']}")
    enhanced_prompt = f"User Query:\n{query_text}\n\nRelevant Code Context:\n" + "\n\n".join(context_texts)
    return enhanced_prompt

# -------------------- GUI FUNCTIONS --------------------
def update_progress(file_done, total_files):
    progress = int((file_done / total_files) * 100)
    progress_var.set(progress)
    status_label.config(text=f"Processed {file_done} of {total_files} files...")
    root.update_idletasks()

def start_embedding_thread():
    repo_url = repo_url_var.get().strip()
    if not repo_url:
        messagebox.showerror("Input Error", "Please enter or select a repository URL.")
        return

    repos = add_repo_to_recent(repo_url)
    repo_dropdown["values"] = repos

    local_repo = clone_or_update_repo(repo_url)
    if not local_repo:
        return

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Processing repository at {local_repo}...\n")
    output_text.update()

    # Get max tokens from the GUI field and compute max_chars (approx. 4 chars per token)
    try:
        max_tokens = int(max_tokens_var.get())
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid max token value: {e}")
        return
    max_chars = max_tokens * 4
    print(f"[DEBUG] Using max_tokens={max_tokens} => max_chars={max_chars}")

    def run_processing():
        global global_embeddings_data
        global_embeddings_data = process_repo_with_progress(local_repo, progress_callback=update_progress, max_chars=max_chars)
        output_text.insert(tk.END, f"\nProcessed {len(global_embeddings_data)} chunks.\n")
        output_file = os.path.join(os.getcwd(), "embedding_output.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(global_embeddings_data, f, indent=2)
            output_text.insert(tk.END, f"\nEmbeddings saved to: {output_file}\n")
        except Exception as e:
            output_text.insert(tk.END, f"\nError saving output: {e}\n")
        progress_var.set(100)
        status_label.config(text="Processing complete.")

    threading.Thread(target=run_processing, daemon=True).start()

def browse_repo():
    folder = filedialog.askdirectory()
    if folder:
        repo_url_var.set(folder)

def generate_prompt_button():
    query = query_prompt_text.get("1.0", tk.END).strip()
    if not query:
        messagebox.showerror("Error", "Please enter a query prompt.")
        return
    if not global_embeddings_data:
        messagebox.showerror("Error", "No embeddings data available. Please process a repository first.")
        return
    enhanced = generate_enhanced_prompt(query, global_embeddings_data, top_k=3)
    if enhanced:
        enhanced_prompt_text.delete("1.0", tk.END)
        enhanced_prompt_text.insert(tk.END, enhanced)
        try:
            with open("enhanced_prompt.txt", "w", encoding="utf-8") as f:
                f.write(enhanced)
        except Exception as e:
            print(f"[DEBUG] Error saving enhanced prompt: {e}")
        messagebox.showinfo("Success", "Enhanced prompt generated!")

# -------------------- SETUP GUI --------------------
root = tk.Tk()
root.title("Codebase Embedding & Retrieval GUI")

# Frame for repository selection
frame_repo = tk.LabelFrame(root, text="Select GitHub Repository URL")
frame_repo.pack(fill=tk.X, padx=10, pady=5)

repo_url_var = tk.StringVar()
recent_repos = load_recent_repos()
repo_dropdown = ttk.Combobox(frame_repo, textvariable=repo_url_var, width=60)
repo_dropdown["values"] = recent_repos
repo_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

browse_btn = tk.Button(frame_repo, text="Browse Folder", command=browse_repo)
browse_btn.pack(side=tk.LEFT, padx=5, pady=5)

# Frame for chunking settings (max chunk size in tokens)
frame_chunk = tk.LabelFrame(root, text="Chunking Settings")
frame_chunk.pack(fill=tk.X, padx=10, pady=5)
tk.Label(frame_chunk, text="Max Chunk Size (tokens):").pack(side=tk.LEFT, padx=5)
max_tokens_var = tk.IntVar(value=128000)
max_tokens_entry = tk.Entry(frame_chunk, textvariable=max_tokens_var, width=10)
max_tokens_entry.pack(side=tk.LEFT, padx=5)

# Progress bar and status label
progress_var = tk.IntVar(value=0)
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5)
status_label = tk.Label(root, text="No processing yet.")
status_label.pack(pady=5)

process_btn = tk.Button(root, text="Process Codebase (Chunk & Embed)", command=start_embedding_thread)
process_btn.pack(pady=10)

# Output text field for process logs
frame_output = tk.LabelFrame(root, text="Embedding Process Output")
frame_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
output_text = tk.Text(frame_output, wrap=tk.WORD, height=15)
output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Frame for query prompt input
frame_query = tk.LabelFrame(root, text="Enter Query Prompt (e.g., Request for code changes or question about the code)")
frame_query.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
query_prompt_text = tk.Text(frame_query, wrap=tk.WORD, height=5)
query_prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Button to generate enhanced prompt
gen_prompt_btn = tk.Button(root, text="Generate Enhanced Prompt", command=generate_prompt_button)
gen_prompt_btn.pack(pady=10)

# Frame for enhanced prompt output
frame_final = tk.LabelFrame(root, text="Enhanced Prompt Output (Copy/Paste this to your AI model)")
frame_final.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
enhanced_prompt_text = tk.Text(frame_final, wrap=tk.WORD, height=15)
enhanced_prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
