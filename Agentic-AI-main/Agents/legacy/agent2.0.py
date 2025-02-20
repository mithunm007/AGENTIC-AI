import ollama
import time
import speech_recognition as sr
from TTS.api import TTS
import sounddevice as sd

# Load a lightweight, GPU-enabled model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

WAKE_WORD = "nexa"


def speak(text):
    """
    Converts the given text to speech using pyttsx3.
    """
    audio = tts.tts(text, return_type="numpy")
    sd.play(audio, samplerate=22050)
    sd.wait()  # Wait until playback finishes

def detect_wake_word():
    """
    Continuously listens for the wake word and then listens for the user's query once detected.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the wake word...")
        while True:
            try:
                # Listen for the wake word with a timeout
                wake_word_audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(wake_word_audio).lower()
                if WAKE_WORD in command:
                    print("Wake word detected! Listening for your query...")
                    
                    # Adjust settings for the query
                    recognizer.pause_threshold = 2.0  # Allow pauses during speech
                    recognizer.dynamic_energy_threshold = True
                    
                    # Listen for the query with an extended time limit
                    query_audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
                    text = recognizer.recognize_google(query_audio)
                    print("You said: " + text)
                    return text
            except sr.UnknownValueError:
                print("Could not understand audio. Waiting for the wake word again...")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                continue

def get_query():
    """
    Listens to the user's query by waiting for the wake word and returns the transcribed text.
    """
    return detect_wake_word()


def process_with_ollama(model, query):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    """
    return ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': query}],
        stream=True,
    )

def handle_response_stream(response):
    """
    Handles the response stream from Ollama, speaking the output in chunks.
    """
    paragraph_buffer = ""  # Initialize a buffer to store paragraph text
    
    for chunk in response:
        print(paragraph_buffer)
        if "." in paragraph_buffer:  
            
            speak(paragraph_buffer)  
            paragraph_buffer = "" 
            
        # print(chunk['message']['content'], end='')  # Print chunk in real-time
        paragraph_buffer += chunk['message']['content']  # Add content to buffer
      

    # Speak any remaining text in the buffer
    if paragraph_buffer:
        speak(paragraph_buffer)
        paragraph_buffer = ""

def calculate_times(start_time, ollama_response_time, speech_generation_time, end_time):
    """
    Calculates and prints the time spent in different sections.
    """
    total_processing_time = end_time - start_time
    ollama_processing_time = ollama_response_time - start_time
    speech_processing_time = speech_generation_time - ollama_response_time

    print("\nTotal processing time:", total_processing_time, "seconds")
    print("Time spent on Ollama response:", ollama_processing_time, "seconds")
    print("Time spent on speech generation:", speech_processing_time, "seconds")

def main():
    """
    Main function to handle the entire speech-to-speech process.
    """
    while True:
        # Step 1: Wait for wake word
        start_time = time.time()

        # Step 2: Listen to query
        query = get_query()

        # Step 3: Process query with Ollama
        response = process_with_ollama('mistral', query)
        ollama_response_time = time.time()

        # Step 4: Handle streamed response and speak it
        handle_response_stream(response)
        speech_generation_time = time.time()

        # Step 5: Calculate and print timings
        end_time = time.time()
        calculate_times(start_time, ollama_response_time, speech_generation_time, end_time)

# Run the main function
if __name__ == "__main__":
    main()
