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

root = None
conversation_log = None
input_box = None
waiting_label = None
model_var = None
token_limit_label = None
prompt_token_label = None

# Tracks the text index where the current streamed reply starts
_current_stream_start = None

# ---------------------------------------------------------------------------
# Real‑time waiting indicator
# ---------------------------------------------------------------------------

def update_waiting_label():
    global waiting, start_time
    if waiting:
        elapsed = time.time() - start_time
        if elapsed < 60:
            waiting_label.config(text=f"Waiting: {int(elapsed)}s")
        else:
            m, s = divmod(int(elapsed), 60)
            waiting_label.config(text=f"Waiting: {m}m {s}s")
        root.after(1000, update_waiting_label)
    else:
        waiting_label.config(text="")

# ---------------------------------------------------------------------------
# Helper that appends streamed chunks to the Text widget as they arrive
# ---------------------------------------------------------------------------

def append_stream_chunk(text: str):
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, text)
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)

# ---------------------------------------------------------------------------
# Finalise streamed answer: add reasoning time and clear waiting flag
# ---------------------------------------------------------------------------

def finalize_stream(reasoning_note: str):
    global waiting
    waiting = False
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"\n\n{reasoning_note}\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)
    waiting_label.config(text="")

# ---------------------------------------------------------------------------
# Network request
# ---------------------------------------------------------------------------

def send_request(user_text: str, dynamic_context: int):
    global waiting, _current_stream_start

    cfg = MODELS_CONFIG[model_var.get()]
    url = f"http://localhost:{cfg['port']}/api/{cfg['endpoint']}"
    payload_field = cfg["payload_field"]
    payload_model = cfg.get("payload_model", model_var.get())
    stream_flag = cfg["stream"]

    # Build payload
    if payload_field == "prompt":
        payload_content = user_text
    elif payload_field == "messages":
        payload_content = [{"role": "user", "content": user_text}]
    else:
        payload_content = user_text

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
            # Insert Bot label once at the beginning of the stream
            root.after(0, lambda: _start_stream_message())
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
                        bot_reply += chunk_text
                        root.after(0, append_stream_chunk, chunk_text)
                    if data.get("done", False):
                        break
            # after stream completes add reasoning note
            reasoning_note = _reasoning_note()
            root.after(0, finalize_stream, reasoning_note)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if payload_field == "messages":
                bot_reply = data.get("message", {}).get("content", "<no content>")
            else:
                bot_reply = data.get("response", "")
            reasoning_note = _reasoning_note()
            root.after(0, _insert_full_bot_message, bot_reply, reasoning_note)
    except requests.exceptions.RequestException as e:
        err = f"Error: {e}"
        reasoning_note = _reasoning_note()
        if stream_flag:
            root.after(0, append_stream_chunk, err)
            root.after(0, finalize_stream, reasoning_note)
        else:
            root.after(0, _insert_full_bot_message, err, reasoning_note)

# ---------------------------------------------------------------------------
# Helper callbacks for GUI insertion
# ---------------------------------------------------------------------------

def _start_stream_message():
    """Insert Bot label once and remember where the text starts."""
    global _current_stream_start
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, "Bot: ", "bot_label")
    _current_stream_start = conversation_log.index(tk.END)
    conversation_log.config(state=tk.DISABLED)


def _insert_full_bot_message(message: str, reasoning_note: str):
    global waiting
    waiting = False
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"Bot: ", "bot_label")
    conversation_log.insert(tk.END, f"\n{message}\n\n{reasoning_note}\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)
    waiting_label.config(text="")


def _reasoning_note() -> str:
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"Reasoned for {int(elapsed)}s."
    m, s = divmod(int(elapsed), 60)
    return f"Reasoned for {m}m {s}s."

# ---------------------------------------------------------------------------
# Send‑button handler
# ---------------------------------------------------------------------------

def send_message(event=None):
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
        _error_to_log(f"Prompt exceeds limit ({allowed_prompt} tokens).")
        return

    dynamic_ctx = 2 * expected_out if token_count <= expected_out else 2 * token_count
    dynamic_ctx = min(dynamic_ctx, cfg_ctx)

    # GUI: show user message
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


def _error_to_log(msg: str):
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"Error: {msg}\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)

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
        return None
    send_message()
    return "break"

# ---------------------------------------------------------------------------
# Build the Tkinter interface
# ---------------------------------------------------------------------------

def create_gui():
    global root, conversation_log, input_box, waiting_label
    global model_var, token_limit_label, prompt_token_label

    root = tk.Tk()
    root.title("QwQ Chat – Streaming Edition")

    # Top bar
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

    # Conversation log
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

    # Input + send
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
