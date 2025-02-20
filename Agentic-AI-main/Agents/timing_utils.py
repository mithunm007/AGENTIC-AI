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
