import tkinter as tk
import requests
import threading
import time
import json
import re  # For processing <think> tags
import os

# Load model configuration from file located in the same directory as this script
def load_model_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "models_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

MODELS_CONFIG = load_model_config()

# Default maximum context window limit (number of tokens) if not specified in config.
DEFAULT_CTX = 8192

def estimate_tokens(text):
    """
    Estimate token count based on a rough approximation of 1 token per 4 characters.
    This is a simple heuristic and may not be exact.
    """
    return max(1, int(len(text) / 4))

# Global variables to control the waiting timer
waiting = False
start_time = None

def update_waiting_label():
    """
    Updates the waiting label with the elapsed time.
    This function schedules itself until waiting is False.
    """
    global waiting, start_time
    if waiting:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            waiting_label.config(text=f"Waiting: {int(elapsed_time)}s")
        else:
            minutes = int(elapsed_time) // 60
            seconds = int(elapsed_time) % 60
            waiting_label.config(text=f"Waiting: {minutes}m {seconds}s")
        # Update every second
        root.after(1000, update_waiting_label)
    else:
        waiting_label.config(text="")

def send_request(user_text, dynamic_context):
    global waiting
    selected_model = model_var.get()
    config = MODELS_CONFIG[selected_model]
    # Use the dynamically computed context limit for this request.
    context_limit = dynamic_context
    bot_reply = ""

    port = config["port"]
    endpoint = config["endpoint"]
    url = f"http://localhost:{port}/api/{endpoint}"
    payload_field = config["payload_field"]
    payload_model = config.get("payload_model", selected_model)
    stream = config["stream"]

    # Determine payload content based on payload_field.
    if payload_field == "prompt":
        payload_content = user_text
    elif payload_field == "messages":
        payload_content = [{"role": "user", "content": user_text}]
    else:
        payload_content = user_text  # Fallback

    payload = {
        "model": payload_model,
        payload_field: payload_content,
        "stream": stream,
        "options": {"num_ctx": context_limit}
    }

    # If quantization is enabled in the config, add it to the payload options.
    if config.get("quantize", False):
        payload["options"]["quantize"] = True

    try:
        if stream:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        chunk_text = data.get("response", "")
                        bot_reply += chunk_text
                        if data.get("done", False):
                            break
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if payload_field == "messages":
                bot_reply = data.get("message", {}).get("content", f"Invalid response received: {response} {data}")
            else:
                bot_reply = data.get("response", "")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error: {str(e)}"

    # Calculate reasoning time and update GUI.
    reasoning_time = time.time() - start_time
    if reasoning_time < 60:
        reasoning_note = f"Reasoned for {int(reasoning_time)}s."
    else:
        minutes = int(reasoning_time) // 60
        seconds = int(reasoning_time) % 60
        reasoning_note = f"Reasoned for {minutes}m {seconds}s."

    root.after(0, update_conversation, bot_reply, reasoning_note)

def insert_bot_message(message):
    """
    Inserts the bot message into the conversation log.  
    For portions of the text enclosed in <think> tags, a toggle button is inserted
    that by default hides the content and allows the user to show/hide it.
    """
    pos = 0
    # Compile a regex that works across newlines
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    
    for match in pattern.finditer(message):
        # Insert normal text (without <think> content) first.
        before_text = message[pos:match.start()]
        if before_text:
            conversation_log.insert(tk.END, before_text)
        
        hidden_content = match.group(1)
        
        # Create a container frame to hold the toggle button and the hidden text.
        container = tk.Frame(conversation_log)
        
        # Create a toggle button. The relief is flat so that it appears like a link.
        toggle_button = tk.Button(container, text="[Show Thought]", relief="flat", borderwidth=0, cursor="hand2")
        toggle_button.pack(anchor="w")
        
        # Create a label for the hidden content. It is styled in a gray italic font.
        # The wraplength is set to 400 pixels (you can adjust this as needed).
        hidden_label = tk.Label(container, text=hidden_content, font=("Helvetica", 10, "italic"), fg="gray", wraplength=400, justify="left")
        hidden_label.pack(anchor="w")
        hidden_label.pack_forget()  # Hide the label initially
        
        # Define the toggle callback function. Using default arguments ensures that each button
        # has its own associated hidden label.
        def toggle(lb=hidden_label, btn=toggle_button):
            if lb.winfo_ismapped():
                lb.pack_forget()
                btn.config(text="[Show Thought]")
            else:
                lb.pack(anchor="w")
                btn.config(text="[Hide Thought]")
        
        toggle_button.config(command=toggle)
        
        # Embed the container (with the button and the hidden label) into the text widget.
        conversation_log.window_create(tk.END, window=container)
        
        pos = match.end()
    
    # Insert any remaining text after the last <think> section.
    if pos < len(message):
        conversation_log.insert(tk.END, message[pos:])

def update_conversation(bot_reply, reasoning_note):
    """
    Updates the conversation log with the bot's reply and reasoning time.
    The bot response is now visually separated from the prompt.
    """
    global waiting
    waiting = False  # Stop the timer
    conversation_log.config(state=tk.NORMAL)
    
    # Insert reasoning time note.
    conversation_log.insert(tk.END, f"{reasoning_note}\n")
    
    # Add extra spacing to visually separate the bot response.
    conversation_log.insert(tk.END, "\n")
    
    # Insert bot label
    conversation_log.insert(tk.END, "Bot: ", "bot_label")
    conversation_log.insert(tk.END, "\n")
    
    # Insert the bot reply with special processing for any <think> blocks.
    insert_bot_message(bot_reply)
    
    conversation_log.insert(tk.END, "\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)  # Auto-scroll to the latest message
    waiting_label.config(text="")  # Clear waiting label

def send_message():
    """
    Called when the user clicks 'Send' or presses Enter (without Shift).
    Retrieves the input, computes a dynamic context limit based on prompt length,
    checks if the prompt is too long, updates the conversation log, starts the waiting timer,
    and launches a thread to perform the network request.
    """
    global waiting, start_time
    user_text = input_box.get("1.0", tk.END).strip()

    if not user_text:
        return

    token_count = estimate_tokens(user_text)
    config_context_limit = MODELS_CONFIG[model_var.get()].get("context_limit", DEFAULT_CTX)
    expected_output = 4096

    # The prompt must fit within (config_context_limit - expected_output)
    allowed_prompt_tokens = config_context_limit - expected_output
    if token_count > allowed_prompt_tokens:
        conversation_log.config(state=tk.NORMAL)
        conversation_log.insert(
            tk.END,
            f"Error: Prompt exceeds the maximum allowed prompt token limit ({allowed_prompt_tokens} tokens) leaving {expected_output} tokens for output.\n\n"
        )
        conversation_log.config(state=tk.DISABLED)
        conversation_log.yview(tk.END)
        return

    # Compute dynamic context limit based on prompt size and expected output.
    # If the prompt is short (<= 4096 tokens), use 8192 tokens.
    # Otherwise, use twice the prompt token count.
    if token_count <= expected_output:
        dynamic_context = 2 * expected_output  # 8192 tokens
    else:
        dynamic_context = 2 * token_count

    # Cap the dynamic context at the model's configured maximum.
    dynamic_context = min(dynamic_context, config_context_limit)

    input_box.delete("1.0", tk.END)
    conversation_log.config(state=tk.NORMAL)
    
    # Insert user prompt with a label.
    conversation_log.insert(tk.END, "You: ", "user_label")
    conversation_log.insert(tk.END, f"{user_text}\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)

    waiting = True
    start_time = time.time()
    update_waiting_label()

    thread = threading.Thread(target=send_request, args=(user_text, dynamic_context))
    thread.start()

def update_token_limit_label(*args):
    """
    Updates the token limit label based on the selected model.
    """
    selected_model = model_var.get()
    context_limit = MODELS_CONFIG.get(selected_model, {}).get("context_limit", DEFAULT_CTX)
    token_limit_label.config(text=f"Token Limit: {context_limit}")

def update_prompt_token_count(event=None):
    """
    Estimates the token count in the prompt field and updates its label.
    """
    text = input_box.get("1.0", tk.END)
    token_count = estimate_tokens(text)
    prompt_token_label.config(text=f"Prompt Tokens: {token_count}")

def handle_enter_key(event):
    """
    Overrides the default behavior of the Return key.
    If Shift is not held, sends the message and prevents a newline.
    If Shift is held, allows a newline.
    """
    # Check if Shift key is pressed (Shift is typically bit 0x0001 in event.state)
    if event.state & 0x0001:
        # Allow Shift+Enter to insert a newline.
        return None
    else:
        send_message()
        # Return "break" to stop the default newline insertion.
        return "break"

def create_gui():
    """
    Sets up the Tkinter GUI with a conversation log, input box, waiting label, send button,
    a dropdown to select the AI model, token limit label, and prompt token count label.
    """
    global root, conversation_log, input_box, waiting_label, model_var, token_limit_label, prompt_token_label

    root = tk.Tk()
    root.title("QwQ Chat")

    top_frame = tk.Frame(root)
    top_frame.pack(padx=10, pady=5, fill=tk.X)

    model_label = tk.Label(top_frame, text="Select AI Model:")
    model_label.pack(side=tk.LEFT)

    # Use model names from the configuration file.
    model_options = list(MODELS_CONFIG.keys())
    model_var = tk.StringVar(root)
    model_var.set(model_options[0])
    model_var.trace("w", update_token_limit_label)

    model_dropdown = tk.OptionMenu(top_frame, model_var, *model_options)
    model_dropdown.pack(side=tk.LEFT, padx=5)

    # Token limit label (based on selected model)
    token_limit_label = tk.Label(top_frame, text="")
    token_limit_label.pack(side=tk.LEFT, padx=5)
    update_token_limit_label()

    frame_log = tk.Frame(root)
    frame_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Conversation log using a Text widget.
    conversation_log = tk.Text(frame_log, wrap=tk.WORD, state=tk.NORMAL)
    conversation_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    conversation_log.tag_configure("user_label", font=("Helvetica", 10, "bold"))
    conversation_log.tag_configure("bot_label", font=("Helvetica", 10, "bold"))
    conversation_log.config(state=tk.DISABLED)

    scrollbar = tk.Scrollbar(frame_log, command=conversation_log.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_log.config(yscrollcommand=scrollbar.set)

    waiting_label = tk.Label(root, text="", font=("Helvetica", 10))
    waiting_label.pack(pady=5)

    # Input box for user text
    input_box = tk.Text(root, height=3, wrap=tk.WORD)
    input_box.pack(padx=10, pady=5, fill=tk.X)
    input_box.bind("<KeyRelease>", update_prompt_token_count)
    # Bind the Return key to our custom handler.
    input_box.bind("<Return>", handle_enter_key)

    # Prompt token count label
    prompt_token_label = tk.Label(root, text="Prompt Tokens: 0")
    prompt_token_label.pack(pady=2)

    send_button = tk.Button(root, text="Send", command=send_message)
    send_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
