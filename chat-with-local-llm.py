import tkinter as tk
import requests
import threading
import time
import json
import re
import os

# ---------------------------------------------------------------------------
# Model configuration helpers
# ---------------------------------------------------------------------------

def load_model_config():
    """Load the JSON model configuration that lives beside this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "models_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

MODELS_CONFIG = load_model_config()
DEFAULT_CTX = 8192  # fallback context length if not in the config

# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Roughly 1 token ≈ 4 chars. Guarantees at least one token."""
    return max(1, len(text) // 4)

# ---------------------------------------------------------------------------
# Globals mutated by the GUI callbacks — keep them at module scope
# ---------------------------------------------------------------------------
waiting: bool = False
start_time: float | None = None

# These will be populated in create_gui()
root = None
conversation_log = None
input_box = None
waiting_label = None
model_var = None
token_limit_label = None
prompt_token_label = None

# ---------------------------------------------------------------------------
# Real‑time waiting indicator
# ---------------------------------------------------------------------------

def update_waiting_label():
    """Refresh the little timer every second while the model is thinking."""
    global waiting, start_time
    if waiting:
        elapsed = time.time() - start_time
        if elapsed < 60:
            waiting_label.config(text=f"Waiting: {int(elapsed)}s")
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            waiting_label.config(text=f"Waiting: {minutes}m {seconds}s")
        root.after(1000, update_waiting_label)
    else:
        waiting_label.config(text="")

# ---------------------------------------------------------------------------
# Helper that appends streamed chunks to the Text widget as they arrive
# ---------------------------------------------------------------------------

def append_stream_chunk(text: str):
    """Thread‑safe GUI append used only for live token streaming."""
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, text)
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)

# ---------------------------------------------------------------------------
# Network request
# ---------------------------------------------------------------------------

def send_request(user_text: str, dynamic_context: int):
    """Send the prompt to the chosen local LLM and stream or batch the reply."""
    global waiting

    selected_model = model_var.get()
    cfg = MODELS_CONFIG[selected_model]

    url = f"http://localhost:{cfg['port']}/api/{cfg['endpoint']}"
    payload_field = cfg["payload_field"]
    payload_model = cfg.get("payload_model", selected_model)
    stream_flag = cfg["stream"]

    # Build request payload
    if payload_field == "prompt":
        payload_content = user_text
    elif payload_field == "messages":
        payload_content = [{"role": "user", "content": user_text}]
    else:
        payload_content = user_text  # fallback

    payload = {
        "model": payload_model,
        payload_field: payload_content,
        "stream": stream_flag,
        "options": {"num_ctx": dynamic_context}
    }
    if cfg.get("quantize", False):
        payload["options"]["quantize"] = True

    bot_reply = ""

    try:
        if stream_flag:
            # ---------------------------------------------------------------
            # Streaming path: iterate over the HTTP chunks and show live text
            # ---------------------------------------------------------------
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_lines(decode_unicode=True):
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    chunk_text = data.get("response", "")
                    if chunk_text:
                        # Live update in the GUI thread
                        root.after(0, append_stream_chunk, chunk_text)
                        bot_reply += chunk_text
                    if data.get("done", False):
                        break
        else:
            # ---------------------
            # Non‑streaming request
            # ---------------------
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if payload_field == "messages":
                bot_reply = data.get("message", {}).get("content", "<no content>")
            else:
                bot_reply = data.get("response", "")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error: {e}"

    # ------------------------------------------
    # Final GUI update once the reply is finished
    # ------------------------------------------
    reasoning_time = time.time() - start_time
    minutes, seconds = divmod(int(reasoning_time), 60)
    reasoning_note = (
        f"Reasoned for {seconds}s." if reasoning_time < 60 else f"Reasoned for {minutes}m {seconds}s."
    )
    root.after(0, update_conversation, bot_reply, reasoning_note)

# ---------------------------------------------------------------------------
# Conversation log helpers
# ---------------------------------------------------------------------------

def insert_bot_message(message: str):
    """Insert bot text; fold <think>...</think> sections behind toggles."""
    pos = 0
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for match in pattern.finditer(message):
        before = message[pos:match.start()]
        if before:
            conversation_log.insert(tk.END, before)

        hidden = match.group(1)
        container = tk.Frame(conversation_log)
        toggle = tk.Button(container, text="[Show Thought]", relief="flat", borderwidth=0, cursor="hand2")
        toggle.pack(anchor="w")
        hidden_lbl = tk.Label(container, text=hidden, font=("Helvetica", 10, "italic"), fg="gray", wraplength=400, justify="left")
        hidden_lbl.pack(anchor="w")
        hidden_lbl.pack_forget()

        def toggle_cb(lb=hidden_lbl, btn=toggle):
            if lb.winfo_ismapped():
                lb.pack_forget(); btn.config(text="[Show Thought]")
            else:
                lb.pack(anchor="w"); btn.config(text="[Hide Thought]")
        toggle.config(command=toggle_cb)
        conversation_log.window_create(tk.END, window=container)
        pos = match.end()

    if pos < len(message):
        conversation_log.insert(tk.END, message[pos:])


def update_conversation(bot_reply: str, reasoning_note: str):
    """Finalise the assistant message and stop the waiting timer."""
    global waiting
    waiting = False

    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"{reasoning_note}\n\n")
    conversation_log.insert(tk.END, "Bot: ", "bot_label")
    conversation_log.insert(tk.END, "\n")
    insert_bot_message(bot_reply)
    conversation_log.insert(tk.END, "\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)
    waiting_label.config(text="")

# ---------------------------------------------------------------------------
# Send‑button handler
# ---------------------------------------------------------------------------

def send_message(event=None):
    """Grab user text, enforce context limits, spawn the worker thread."""
    global waiting, start_time

    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        return

    token_count = estimate_tokens(user_text)
    cfg = MODELS_CONFIG[model_var.get()]
    cfg_ctx = cfg.get("context_limit", DEFAULT_CTX)
    expected_out = 4096

    allowed_prompt = cfg_ctx - expected_out
    if token_count > allowed_prompt:
        conversation_log.config(state=tk.NORMAL)
        conversation_log.insert(tk.END, f"Error: Prompt exceeds limit ({allowed_prompt} tokens).\n\n")
        conversation_log.config(state=tk.DISABLED)
        conversation_log.yview(tk.END)
        return

    dynamic_ctx = 2 * expected_out if token_count <= expected_out else 2 * token_count
    dynamic_ctx = min(dynamic_ctx, cfg_ctx)

    # Update GUI with user message
    input_box.delete("1.0", tk.END)
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, "You: ", "user_label")
    conversation_log.insert(tk.END, f"{user_text}\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)

    waiting = True
    start_time = time.time()
    update_waiting_label()

    threading.Thread(target=send_request, args=(user_text, dynamic_ctx), daemon=True).start()

# ---------------------------------------------------------------------------
# GUI helpers — token limit and prompt token counter
# ---------------------------------------------------------------------------

def update_token_limit_label(*_):
    ctx = MODELS_CONFIG.get(model_var.get(), {}).get("context_limit", DEFAULT_CTX)
    token_limit_label.config(text=f"Token Limit: {ctx}")


def update_prompt_token_count(event=None):
    count = estimate_tokens(input_box.get("1.0", tk.END))
    prompt_token_label.config(text=f"Prompt Tokens: {count}")

# ---------------------------------------------------------------------------
# <Shift>‑Enter vs Enter behaviour
# ---------------------------------------------------------------------------

def handle_enter_key(event):
    if event.state & 0x0001:  # Shift pressed
        return None  # allow newline
    send_message()
    return "break"  # suppress default newline

# ---------------------------------------------------------------------------
# Build the Tkinter interface
# ---------------------------------------------------------------------------

def create_gui():
    global root, conversation_log, input_box, waiting_label
    global model_var, token_limit_label, prompt_token_label

    root = tk.Tk()
    root.title("QwQ Chat – Streaming Edition")

    # Top bar: model selector & token limit
    top = tk.Frame(root)
    top.pack(padx=10, pady=5, fill=tk.X)

    tk.Label(top, text="Select AI Model:").pack(side=tk.LEFT)
    model_options = list(MODELS_CONFIG.keys())
    model_var = tk.StringVar(root, value=model_options[0])
    model_var.trace("w", update_token_limit_label)
    tk.OptionMenu(top, model_var, *model_options).pack(side=tk.LEFT, padx=5)

    token_limit_label = tk.Label(top, text="")
    token_limit_label.pack(side=tk.LEFT, padx=5)
    update_token_limit_label()

    # Conversation log + scrollbar
    frame_log = tk.Frame(root)
    frame_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    conversation_log = tk.Text(frame_log, wrap=tk.WORD, state=tk.NORMAL)
    conversation_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    conversation_log.tag_config("user_label", font=("Helvetica", 10, "bold"))
    conversation_log.tag_config("bot_label", font=("Helvetica", 10, "bold"))
    conversation_log.config(state=tk.DISABLED)

    scrollbar = tk.Scrollbar(frame_log, command=conversation_log.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_log.config(yscrollcommand=scrollbar.set)

    waiting_label = tk.Label(root, text="", font=("Helvetica", 10))
    waiting_label.pack(pady=5)

    # Input + token counter + send
    input_box = tk.Text(root, height=3, wrap=tk.WORD)
    input_box.pack(padx=10, pady=5, fill=tk.X)
    input_box.bind("<KeyRelease>", update_prompt_token_count)
    input_box.bind("<Return>", handle_enter_key)

    prompt_token_label = tk.Label(root, text="Prompt Tokens: 0")
    prompt_token_label.pack(pady=2)

    tk.Button(root, text="Send", command=send_message).pack(pady=5)

    root.mainloop()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    create_gui()
