import tkinter as tk
import requests
import threading
import time
import re  # For processing <think> tags

# Global variables to control the waiting timer
waiting = False
start_time = None

# Maximum context window limit (number of tokens)
MAX_CTX = 8192

def estimate_tokens(text):
    """
    Estimate token count based on a rough approximation of 1 token per 4 characters.
    This is a simple heuristic and may not be exact.
    """
    return max(1, int(len(text) / 4))

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

def send_request(user_text):
    """
    Runs in a separate thread to call the Ollama API and fetch the response.
    When done, it schedules an update to the conversation log on the main thread.
    """
    global waiting
    # Choose the port based on the selected model
    selected_model = model_var.get()
    if selected_model == "deepseek-r1:7b":
        port = 11437
    else:
        port = 11434

    url = f"http://localhost:{port}/api/chat"
    try:
        response = requests.post(
            url,
            json={
                "model": selected_model,
                "messages": [{"role": "user", "content": user_text}],
                "stream": False,
                "options": {
                    "num_ctx": MAX_CTX  # Set the context window token limit
                }
            }
            # No timeout parameter to allow indefinite waiting
        )
        response.raise_for_status()
        data = response.json()
        bot_reply = data.get("message", {}).get("content", "<No response received>")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error: {str(e)}"
    
    # Calculate reasoning time
    reasoning_time = time.time() - start_time
    if reasoning_time < 60:
        reasoning_note = f"Reasoned for {int(reasoning_time)}s."
    else:
        minutes = int(reasoning_time) // 60
        seconds = int(reasoning_time) % 60
        reasoning_note = f"Reasoned for {minutes}m {seconds}s."
    
    # Update the GUI in the main thread
    root.after(0, update_conversation, bot_reply, reasoning_note)

def insert_bot_message(message):
    """
    Inserts the bot message into the conversation log,
    formatting <think></think> parts differently.
    """
    pos = 0
    for match in re.finditer(r"<think>(.*?)</think>", message):
        # Insert text before the <think> block
        if match.start() > pos:
            conversation_log.insert(tk.END, message[pos:match.start()])
        # Insert the <think> block in italic gray text
        conversation_log.insert(tk.END, match.group(1), "think")
        pos = match.end()
    # Insert any remaining text after the last <think> block
    if pos < len(message):
        conversation_log.insert(tk.END, message[pos:])

def update_conversation(bot_reply, reasoning_note):
    """
    Updates the conversation log with the bot's reply and reasoning time, then stops the timer.
    """
    global waiting
    waiting = False  # Stop the timer
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"{reasoning_note}\n")
    conversation_log.insert(tk.END, "Bot: ", "bot_label")
    insert_bot_message(bot_reply)
    conversation_log.insert(tk.END, "\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)  # Auto-scroll to the latest message
    waiting_label.config(text="")  # Clear waiting label

def send_message():
    """
    Called when the user clicks 'Send'. It retrieves the input,
    checks if it exceeds the context token limit, updates the conversation log,
    starts the waiting timer, and launches a thread to perform the network request.
    """
    global waiting, start_time
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        return  # Do nothing if the input is empty

    # Check if the estimated token count exceeds the maximum context window
    if estimate_tokens(user_text) > MAX_CTX:
        conversation_log.config(state=tk.NORMAL)
        conversation_log.insert(tk.END, f"Error: Prompt exceeds the maximum allowed token limit ({MAX_CTX} tokens). Please shorten your input.\n\n")
        conversation_log.config(state=tk.DISABLED)
        conversation_log.yview(tk.END)
        return

    # Clear the input box for the next message
    input_box.delete("1.0", tk.END)

    # Display user's message in the conversation log with bold "You:" label
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, "You: ", "user_label")
    conversation_log.insert(tk.END, f"{user_text}\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)  # Auto-scroll to the latest message

    # Start waiting timer
    waiting = True
    start_time = time.time()
    update_waiting_label()

    # Start the network call in a separate thread
    thread = threading.Thread(target=send_request, args=(user_text,))
    thread.start()

def create_gui():
    """
    Sets up the Tkinter GUI with a conversation log, input box, waiting label, send button,
    and a dropdown to select the AI model.
    """
    global root, conversation_log, input_box, waiting_label, model_var
    root = tk.Tk()
    root.title("QwQ Chat")

    # --- Top frame for model selection ---
    top_frame = tk.Frame(root)
    top_frame.pack(padx=10, pady=5, fill=tk.X)

    model_label = tk.Label(top_frame, text="Select AI Model:")
    model_label.pack(side=tk.LEFT)

    # Define available models; default is deepseek-r1:7b and second option is qwq
    model_options = ["deepseek-r1:7b", "qwq"]
    model_var = tk.StringVar(root)
    model_var.set(model_options[0])  # Default model is deepseek-r1:7b

    model_dropdown = tk.OptionMenu(top_frame, model_var, *model_options)
    model_dropdown.pack(side=tk.LEFT, padx=5)

    # --- Frame for conversation log ---
    frame_log = tk.Frame(root)
    frame_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    conversation_log = tk.Text(frame_log, wrap=tk.WORD, state=tk.NORMAL)
    conversation_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure text tags for formatting
    conversation_log.tag_configure("user_label", font=("Helvetica", 10, "bold"))
    conversation_log.tag_configure("bot_label", font=("Helvetica", 10, "bold"))
    conversation_log.tag_configure("think", foreground="gray", font=("Helvetica", 10, "italic"))
    conversation_log.config(state=tk.DISABLED)

    scrollbar = tk.Scrollbar(frame_log, command=conversation_log.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_log.config(yscrollcommand=scrollbar.set)

    waiting_label = tk.Label(root, text="", font=("Helvetica", 10))
    waiting_label.pack(pady=5)

    input_box = tk.Text(root, height=3, wrap=tk.WORD)
    input_box.pack(padx=10, pady=5, fill=tk.X)

    send_button = tk.Button(root, text="Send", command=send_message)
    send_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
