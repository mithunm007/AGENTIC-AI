import os
from langchain_groq import ChatGroq
import ollama
from speech_utils import speak

def process_with_ollama(model="mistral", query=''):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    """
    # system_prompt = """
    #             You are an orchestrator responsible for managing conversations with the user.  

    #             - If the user asks a question you can answer, respond directly and concisely.  
    #             - If the user asks a question you cannot answer, inform them that you will delegate the query to an agent. Politely ask the user to wait while the agent processes the request.
    #             - If user asks about anything with keywords like new or latest. you must not answer and say u will delegate it  

    #             During the agent's processing time:  
    #             - Continue the conversation with the user to maintain engagement.  
    #             - Do not provide any additional information about the agent's actions or the process.
    #             - Do not provide any other details other than what the user exactly asked for   

    #             Important Guidelines:  
    #             1. Only respond to user queries if you are confident in your knowledge.  
    #             2. For unknown queries, delegate them to the agent and wait for the agent's output.  
    #             3. Do not provide unnecessary explanations or speculative responses.  

    #             The agent will automatically handle and return the output to the user. Your role is to keep the conversation natural and engaging without overstepping these boundaries.
                
    #             """
    
    system_prompt = """
                    you are a conversational assistant. you will converse with the user keep your conversation concise, clean and short.    
    """
    return ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}],
        stream=True,
    )
    
def process_with_groq(query=''):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    input format:
    message =  [
        (
            "system",
            "You are an intent recognition model",
        ),
        ("human", instructions),
    ]
    """
    
    llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768"
    )
    groq_response = llm.invoke(query)
    
    return groq_response.content



    
def handle_response_stream(response):
    """
    Handles the response stream from Ollama, speaking the output in chunks.
    """
    paragraph_buffer = ""  # Initialize a buffer to store paragraph text
    
    for chunk in response:
        
        #make sure to yield the content of the message when running in gradio
        
        # yield chunk['message']['content']
        if ("." in paragraph_buffer) and (len(paragraph_buffer) > 15):  
            print(paragraph_buffer)
            speak(paragraph_buffer)  
            paragraph_buffer = "" 
        
        paragraph_buffer += chunk['message']['content']  # Add content to buffer

    # Speak any remaining text in the buffer
    if paragraph_buffer:
        print(paragraph_buffer)
        speak(paragraph_buffer)
        paragraph_buffer = ""


if __name__ == '__main__':
    '''res = process_with_groq(query="""you will act as an intent recognizer.
    Your can only reply with one word and no more than that:
    user query = "look up on the internet for 5 best phones" 
    1. If the user query is refering anything about searching on the internet then you will reply with the word = 'internetSearch'
    2. If the user query is asking for news then you will reply with the word = 'newsSearch'
    3. if anything else you will reply with the word = 'NO'
    """
    )
    print(res)'''
    res = process_with_ollama(query="What is the capital of France?")
    print(handle_response_stream(res))