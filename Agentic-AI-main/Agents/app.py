import os
import gradio as gr
import time
from crew_ai.all_agents import extract_pdf_information, extract_text_information
from speech_utils import detect_wake_word, speak
from agent_utils import dispatch_agent
from ollama_utils import process_with_ollama, handle_response_stream
from timing_utils import calculate_times
import shutil


# Define the storage path for uploaded files
UPLOAD_DIR = "./crew_ai/files/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_file(file):
    """
    Save the uploaded file from Gradio's File component to the specified directory.

    Args:
        file: The uploaded file object.

    Returns:
        str: The path to the saved file, or None if no file is provided.
    """
    if not file:
        return None

    # Gradio provides a temporary path via `file.name`
    source_path = file.name  # The temporary file path
    file_name = os.path.basename(source_path)  # Extract the original file name
    target_path = os.path.join(UPLOAD_DIR, file_name)  # Target path in UPLOAD_DIR

    # Copy the file to the target directory
    shutil.copy(source_path, target_path)

    return target_path


def process_file_query(file, question, intent):
    """
    Handle file upload and question input for specialized intents.
    """
    if not file or not question:
        return "Please provide both a file and a question."

    # Save the file
    file_path = save_file(file)
    if not file_path:
        return "Error saving the uploaded file."

    # Process based on intent
    if intent == "pdfsearch":
        return extract_pdf_information(file_path, question)
    elif intent == "textsearch":
        return extract_text_information(file_path, question)
    else:
        return "Invalid intent."


def process_query_stream(query, use_voice):
    """
    Processes the query and yields responses in chunks.
    """
    start_time = time.time()

    if use_voice and not query:
        query = detect_wake_word()
        if not query:
            yield "Listening for wake word..."
            return

    if query:
        response = dispatch_agent(query)
        if "Sorry" in response or "Error" in response:
            response_stream = process_with_ollama("mistral", query)
            accumulated_response = ""

            for partial_response in handle_response_stream(response_stream):
                accumulated_response += partial_response
                yield accumulated_response
        else:
            yield response
            speak(response)

        calculate_times(start_time, time.time(), time.time(), time.time())
        return

    yield "No input received."


# Gradio App with Chat-Like Interface and File Upload
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 NEXUS AI Assistant - Voice & Text Interface")

    with gr.Row():  # Create a row for left and right sections
        with gr.Column(scale=3):  # Left side (larger width)
            use_voice = gr.Checkbox(label="Enable Voice Mode", value=False)
            chatbot = gr.Chatbot(label="Conversation", height=300)  # Adjust height as needed
            query_box = gr.Textbox(
                label="Enter your query or enable Voice Mode.", 
                lines=1, 
                placeholder="Type your query here...",
                max_lines=3,
            )
            submit_button = gr.Button("Submit")

        with gr.Column(scale=1):  # Right side (smaller width for file upload)
            file_upload = gr.File(label="Upload File (PDF/Text)")
            question_box = gr.Textbox(
                label="Enter Your Question", 
                placeholder="Type your question here...", 
                lines=2
            )
            intent_dropdown = gr.Dropdown(
                choices=["pdfsearch", "textsearch"], label="Select Search Intent"
            )
            file_process_button = gr.Button("Process File & Question")

    # Function to handle the chat logic
    def chatbot_response(chat_history, user_input, use_voice):
        """
        Update the chat history as responses are streamed.
        """
        if use_voice and not user_input:
            user_input = detect_wake_word()

        if not user_input:
            return chat_history + [("No input received.", None)], ""

        # Add the user's message to the left
        chat_history.append((user_input, None))

        # Stream assistant responses
        accumulated_response = ""
        chat_history.append((None, ""))  # Placeholder for the assistant's response

        for response in process_query_stream(user_input, use_voice):
            accumulated_response = response
            chat_history[-1] = (None, accumulated_response)  # Update assistant's response
            yield chat_history, ""

        # Clear the query box after submission
        return chat_history, ""

    # Function to handle file and question logic
    def handle_file_and_question(chat_history, file, question, intent):
        """
        Process the file and question and update the chat history.
        """
        if not file or not question:
            response = "Please provide both a file and a question."
        else:
            response = process_file_query(file, question, intent)
            
        speak(response)

        # Add the user's input (left)
        chat_history.append((f"File uploaded\nQuestion: {question}", None))

        # Add the assistant's response (right)
        chat_history.append((None, response))

        # Clear the question box after submission
        return chat_history, ""

    # Connect components
    query_box.submit(  # Trigger submission on pressing Enter in the query_box
        fn=chatbot_response,
        inputs=[chatbot, query_box, use_voice],
        outputs=[chatbot, query_box],  # Clear the query_box
    )

    submit_button.click(  # Trigger submission on button click
        fn=chatbot_response,
        inputs=[chatbot, query_box, use_voice],
        outputs=[chatbot, query_box],  # Clear the query_box
    )

    file_process_button.click(
        fn=handle_file_and_question,
        inputs=[chatbot, file_upload, question_box, intent_dropdown],
        outputs=[chatbot, question_box],  # Clear the question_box
    )

demo.launch()

