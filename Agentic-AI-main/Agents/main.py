import time
from speech_utils import detect_wake_word, speak
from agent_utils import dispatch_agent
from ollama_utils import process_with_groq, process_with_ollama, handle_response_stream
from timing_utils import calculate_times
import streamlit as st


def main():
    """
    Main function to handle the entire speech-to-speech process.
    """
    while True:
        try:
            # Step 1: Wait for wake word and listen to query
            start_time = time.time()
            
            query = detect_wake_word()
            
            if query:
                # Step 2: Determine and dispatch agent
                print("Processing query with intent recognition...")
                response = dispatch_agent(query)
                print("please wait while I process the query.")

                if "Sorry" in response or "Error" in response:
                    # If no valid agent matched, fallback to Ollama processing
                    print("Fallback to general query processing...")
                    
                    
                    response =  process_with_ollama('mistral', query)
                    handle_response_stream(response)
                else:
                    # instructions = f"""
                    # You are an orchestrator responsible for refining outputs from multiple agents. 
                    # Your task is to process the given agent output and return a response that is clean, concise, and user-friendly.

                    # Agent Output: {response}

                    # Requirements:
                    # 1. Only return your refined and cleaned output.
                    # 2. Do not include the raw agent output or any unnecessary details in your response.
                    # 3. Ensure the response is clear, well-structured, and directly addresses the context of the agent's output.

                    # Your sole responsibility is to provide a polished and professional response based on the agent's output.
                    # """
                    
                    # llm_response = process_with_ollama(instructions)
                    print(response)
                    speak(response)  # Speak the refined and cleaned response chunk by chunk   

                # Step 3: Timings
                end_time = time.time()
                calculate_times(start_time, time.time(), time.time(), end_time)

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break

if __name__ == "__main__":
    main()