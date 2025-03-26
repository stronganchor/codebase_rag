import tkinter as tk
import requests
import threading

# Global variables to control the waiting animation
waiting = False
animation_index = 0

def animate_status():
    """
    Updates the waiting label with an ellipsis animation.
    This function schedules itself until waiting is False.
    """
    global waiting, animation_index
    if waiting:
        states = ["", ".", "..", "..."]
        waiting_label.config(text="Waiting" + states[animation_index % len(states)])
        animation_index += 1
        # Update every 500 ms
        root.after(500, animate_status)
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
        )
        response.raise_for_status()
        data = response.json()
        bot_reply = data.get("message", {}).get("content", "<No response received>")
    except requests.exceptions.RequestException as e:
        bot_reply = f"Error: {str(e)}"
    
    # Update the GUI in the main thread
    root.after(0, update_conversation, bot_reply)

def update_conversation(bot_reply):
    """
    Updates the conversation log with the bot's reply and stops the animation.
    """
    global waiting
    waiting = False  # Stop the animation
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"Bot: {bot_reply}\n\n")
    conversation_log.config(state=tk.DISABLED)
    waiting_label.config(text="")  # Clear waiting label

def send_message():
    """
    Called when the user clicks 'Send'. It retrieves the input,
    updates the conversation log, starts the waiting animation,
    and launches a thread to perform the network request.
    """
    global waiting, animation_index
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        return  # Do nothing if the input is empty

    # Clear the input box for the next message
    input_box.delete("1.0", tk.END)

    # Display user's message in the conversation log
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, f"You: {user_text}\n")
    conversation_log.config(state=tk.DISABLED)

    # Start waiting animation
    waiting = True
    animation_index = 0
    animate_status()

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

    # Frame for the conversation log
    frame_log = tk.Frame(root)
    frame_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Text widget for conversation log (read-only)
    conversation_log = tk.Text(frame_log, wrap=tk.WORD, state=tk.DISABLED)
    conversation_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Scrollbar for conversation log
    scrollbar = tk.Scrollbar(frame_log, command=conversation_log.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_log.config(yscrollcommand=scrollbar.set)

    # Waiting label for animation
    waiting_label = tk.Label(root, text="", font=("Helvetica", 10))
    waiting_label.pack(pady=5)

    # Input box for user text
    input_box = tk.Text(root, height=3, wrap=tk.WORD)
    input_box.pack(padx=10, pady=5, fill=tk.X)

    # Send button
    send_button = tk.Button(root, text="Send", command=send_message)
    send_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
