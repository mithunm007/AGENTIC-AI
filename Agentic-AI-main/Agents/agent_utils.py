
from crew_ai.all_agents import fetch_latest_news, fetch_internet_search_results,extract_pdf_information,extract_text_information,fetch_youtube_video_data
from ollama_utils import handle_response_stream, process_with_groq, process_with_ollama


def check_agent_call(text):
    """
    Determine which agent to call based on the query.
    """
    instructions = f"""
                You are an intent recognizer. Your *ONLY* output must be a single word chosen from the list below. Provide *NO* explanations, punctuation, or additional text.

                **Input:** User query: {text}

                Your purpose is to identify whether the user's query requires specialized actions beyond the capabilities of a standard LLM. For any general query answerable by an LLM, respond with **NO**. For specialized tasks like those listed below, respond with the corresponding single word.

                **Possible Outputs (Choose ONE):**
                1. **internetSearch:** For general internet searches.
                2. **newsSearch:** For news or latest updates.
                3. **pdfSearch:** For PDF files or content within them.
                4. **textSearch:** For text files or searching within them.
                5. **youtubeSearch:** For videos, tutorials, or learning resources.
                6. **NO:** For all other queries.

                **RULES:**
                1. **ONE WORD ONLY:** Respond with a single word from the list above.
                2. **NO EXPLANATIONS:** Do not add any extra text or reasoning.
                3. **NO PUNCTUATION:** Avoid all punctuation marks.
                4. **CASE SENSITIVE:** Match the case of the words as given (e.g., "internetSearch").

                **Examples:**
                - **Input:** "Find the latest iPhone reviews."
                **Output:** internetSearch
                - **Input:** "Explain quantum physics."
                **Output:** NO
                - **Input:** "Summarize this PDF file."
                **Output:** pdfSearch

                """
    message =  [
        (
            "system",
            "You are an intent recognition model",
        ),
        ("human", instructions),
    ]
    result = process_with_groq(query=message)
    print(result)
    return result

def dispatch_agent(user_query):
    """
    Map the intent to agents and execute the appropriate action.
    """
    intent = check_agent_call(user_query).lower()
    
    if "internetsearch" in intent:
        print("Calling Internet Search Agent...")
        return fetch_internet_search_results(user_query)
    
    elif "newssearch" in intent:
        print("Calling News Search Agent...")
        return fetch_latest_news()
    
    # elif "pdfsearch" in intent:
    #     print("Calling PDF Search Agent...")
    #     pdf_path=input("please provide the PDF file: ").strip()
    #     question=input("What is the topic to search for: ").strip()
    #     return extract_pdf_information(pdf_path, question)
    
    # elif "textsearch" in intent:        
    #     print("Calling Text Search Agent...")
    #     txt_path=input("please provide the Text file: ").strip()
    #     question=input("What is the topic to search for: ").strip()
    #     return extract_text_information(txt_path, question)
    
    elif "youtubesearch" in intent:
        print("Calling YouTube Search Agent...")
        # yt_output = 
        # query = yt_output + "/n this is the result from Youtube api, process and summerize it and response to user that they can refer this video."
        # final_res = process_with_ollama(query)
        return fetch_youtube_video_data(user_query)
        
    elif intent == "no":
        return "Sorry, I cannot process your request. Please rephrase or clarify your query."
    
    else:
        return "Error: Unknown intent returned by the intent recognizer."


if __name__ == "__main__":
    user_query = "best video to learn assembly"
    response = dispatch_agent(user_query)
    print(response)