import tkinter as tk
from tkinter import font as tkFont
import requests
import threading
import time
import re

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

def send_request(user_text):
    """
    Runs in a separate thread to call the Ollama API and fetch the response.
    When done, it schedules an update to the conversation log on the main thread.
    """
    global waiting
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwq",  # Ensure this matches your Ollama model name
                "messages": [{"role": "user", "content": user_text}],
                "stream": False
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

def update_conversation(bot_reply, reasoning_note):
    """
    Updates the conversation log with the bot's reply and reasoning time, then stops the timer.
    """
    global waiting
    waiting = False  # Stop the timer
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"{reasoning_note}\n", 'regular')
    conversation_log.insert(tk.END, "Bot: ", 'bold')
    
    # Process the bot's reply to handle <think> tags
    parts = re.split(r'(<think>|</think>)', bot_reply)
    italic = False
    for part in parts:
        if part == '<think>':
            italic = True
        elif part == '</think>':
            italic = False
        else:
            tag = 'think' if italic else 'regular'
            conversation_log.insert(tk.END, part, tag)
    
    conversation_log.insert(tk.END, "\n\n")
    conversation_log.config(state=tk.DISABLED)
    conversation_log.yview(tk.END)  # Auto-scroll to the latest message
    waiting_label.config(text="")  # Clear waiting label

def send_message():
    """
    Called when the user clicks 'Send'. It retrieves the input,
    updates the conversation log, starts the waiting timer,
    and launches a thread to perform the network request.
    """
    global waiting, start_time
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        return  # Do nothing if the input is empty

    # Clear the input box for the next message
    input_box.delete("1.0", tk.END)

    # Display user's message in the conversation log
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, "You: ", 'bold')
    conversation_log.insert(tk.END, f"{user_text}\n", 'regular')
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
    Sets up the Tkinter GUI with a conversation log, input box, waiting label, and send button.
    """
    global root, conversation_log, input_box, waiting_label
    root = tk.Tk()
    root.title("QwQ Chat")

    # Define fonts
    bold_font = tkFont.Font(root=root, family="Helvetica", weight="bold")
    italic_font = tkFont.Font(root=root, family="Helvetica", slant="italic")

    # Frame for the conversation log
    frame_log = tk.Frame(root)
    frame_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Text widget for conversation log (read-only)
    conversation_log = tk.Text(frame_log, wrap=tk.WORD, state=tk.DISABLED)
    conversation_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a vertical scrollbar for the conversation log
    scrollbar = tk.Scrollbar(frame_log, command=conversation_log.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_log.config(yscrollcommand=scrollbar.set)

    # Define tags for text formatting
    conversation_log.tag_configure('bold', font=bold_font)
    conversation_log.tag_configure('think', font=italic
::contentReference[oaicite:13]{index=13}
 
