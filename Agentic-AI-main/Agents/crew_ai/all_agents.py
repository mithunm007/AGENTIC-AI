import os
from crewai import Agent, Crew, Task
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from googleapiclient.discovery import build
from crewai_tools import SerperDevTool , PDFSearchTool , TXTSearchTool

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/mixtral-8x7b-32768"
)

load_dotenv()

def fetch_internet_search_results(topic):
    """
    Function to perform an internet search for a given topic using CrewAI and SerperDevTool.
    Returns a summarized result of the query along with relevant links.

    Args:
        topic (str): The search query or topic for the internet search.

    Returns:
        str: A summarized report of the search results.
    """
    # Load environment variables

    # Define the Internet Search Agent
    internet_search_agent = Agent(
        role="Internet Search Agent",
        goal=""" 
            Your task is to find information on the internet using the 'Search the internet' tool. 
            Use it to execute a search query based on the user's input, evaluate the credibility 
            of the returned results, and provide a clear summary of the information.
        """,
        backstory=""" 
            You are a specialized agent for web searches, extracting reliable and concise information 
            from the internet to address user queries.
        """,
        verbose=False,
        llm=llm,
    )

    # Initialize the InternetSearchTool
    search_tool = SerperDevTool(
        n_results=2,
    )

    # Define the Task
    internet_search_task = Task(
        description=f""" 
            Use the 'search_tool' tool to find information about the topic '{topic}'.
            You should:
            1. Review the results returned by the tool.
            2. Summarize the findings along with the links in a concise and clear manner.
        """,
        expected_output=""" 
            A summarized report of relevant information gathered about the topic.
        """,
        agent=internet_search_agent,
        tools=[search_tool],
    )

    # Crew Definition
    crew = Crew(
        agents=[internet_search_agent],
        tasks=[internet_search_task],
        verbose=False,
    )

    # Execute the Task

    result = crew.kickoff(inputs={"topic": topic})
    return str(result)





def fetch_latest_news():
    """
    Function to perform a news search for a given topic using CrewAI and SerperDevTool.
    Returns a summarized result of the news query along with relevant links.

    Args:
        topic (str): The search query or topic for the news search.

    Returns:
        str: A summarized report of the latest news gathered about the topic.
    """
    # Load environment variables

    # Define the LLM
    # llm = ChatOpenAI(
    #     model="ollama/mistral",
    #     base_url="http://localhost:11434"
    # )

    topic = "lastest news from all over the world"
    
    # Define the News Search Agent
    news_search_agent = Agent(
        role="News Search Agent",
        goal=""" 
            Your task is to find information on the internet using the 'search_tool' tool. 
            Use it to execute a search query based on the user's input, evaluate the credibility 
            of the returned results, and provide a clear summary of the information.
        """,
        backstory=""" 
            You are a specialized agent for web searches, extracting reliable and concise news information 
            from the internet to address user queries.
        """,
        verbose=False,
        llm=llm,
    )

    # Initialize the InternetSearchTool
    search_tool = SerperDevTool(
        n_results=3,
    )

    # Define the Task
    internet_search_task = Task(
        description=f""" 
            Use the 'search_tool' tool to find information about the latest news on the internet.
            use the tool and input the search query {topic} to the tool and gather the news
            You should:
            1. Be sure to pass the same query to the tool dont change anything. Review the results returned by the tool.
            2. Summarize the findings along with the links in a concise and clear manner.
        """,
        expected_output=""" 
            A summarized report of relevant news.
        """,
        agent=news_search_agent,
        tools=[search_tool],
    )

    # Crew Definition
    crew = Crew(
        agents=[news_search_agent],
        tasks=[internet_search_task],
        verbose=False,
    )

    # Execute the Task
    result = crew.kickoff(inputs={"topic": topic})
    return str(result)


def extract_pdf_information(pdf_path, question):
    """
    Function to retrieve and refine information from a PDF file using CrewAI and PDFSearchTool.
    Returns a concise and contextually aligned result.

    Args:
        pdf_path (str): The path to the PDF file.
        question (str): The query to retrieve information about from the PDF.

    Returns:
        str: Refined text containing only the information relevant to the question.
    """
    # Load environment variables

    # # Define the LLM
    # llm = ChatOpenAI(
    #     model="ollama/mistral",
    #     base_url="http://localhost:11434"
    # )

    
    
    # Define the PDF Search Agent
    pdf_search_agent = Agent(
        role="PDF Search Agent",
        goal=""" 
            Act as a document retrieval and refinement agent. Your task is to process raw 
            PDF files and extract relevant information based on the user's query. 
            Ensure the output is concise, accurate, and directly aligned with the query.
        """,
        backstory=""" 
            You are a specialized document analysis agent, trained to efficiently retrieve 
            and refine information from large PDF files. Your expertise lies in presenting 
            information clearly and accurately based on user queries.
        """,
        verbose=False,
        llm=llm,
    )

    # Initialize the PDFSearchTool
    search_tool = PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="mixtral-8x7b-32768",
                    temperature=0.1,
                    top_p=1,
                    stream=True,
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

    # Define the Task
    pdf_retrieval_task = Task(
        description=f""" 
            Use the PDFSearchTool to process the PDF file of name {pdf_path} and 
            extract information relevant to the query: '{question}' from that pdf.
            Ensure the output is concise, accurate, and contextually aligned with the query.
            dont explain how u did it. just give the output and NOTHING else.
        """,
        expected_output=""" 
            Refined text containing only the information relevant to the question.
        """,
        agent=pdf_search_agent,
        tools=[search_tool],
    )

    # Crew Definition
    crew = Crew(
        agents=[pdf_search_agent],
        tasks=[pdf_retrieval_task],
        verbose=False,
    )

    # Execute the Task
    result = crew.kickoff(inputs={"question": question})
    return str(result)



def extract_text_information(txt_path, question):
    """
    Function to retrieve and refine information from a text file using CrewAI and TXTSearchTool.
    Returns a concise and contextually aligned result.

    Args:
        txt_path (str): The path to the text file.
        question (str): The query to retrieve information about from the text.

    Returns:
        str: Refined text containing only the information relevant to the question.
    """
    # Load environment variables

    # # Define the LLM
    # llm = ChatOpenAI(
    #     model="ollama/mistral",
    #     base_url="http://localhost:11434"
    # )


    
    
    # Define the Text Search Agent
    text_search_agent = Agent(
        role="Text Search Agent",
        goal=""" 
            Act as a text retrieval and refinement agent. Your task is to process raw 
            text files and extract relevant information based on the user's query. 
            Ensure the output is concise, accurate, and directly aligned with the query.
        """,
        backstory=""" 
            You are a specialized text analysis agent, trained to efficiently retrieve 
            and refine information from large text files. Your expertise lies in presenting 
            information clearly and accurately based on user queries.
        """,
        verbose=False,
        llm=llm,
    )

    # Initialize the TXTSearchTool
    search_tool = TXTSearchTool(
        txt=txt_path,
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="mixtral-8x7b-32768",
                    temperature=0.1,
                    top_p=1,
                    stream=True,
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

    # Define the Task
    text_retrieval_task = Task(
        description=f""" 
            Use the TXTSearchTool to process the text file and 
            extract information relevant to the query: '{question}'. 
            Ensure the output is concise, accurate, and contextually aligned with the query.
            dont explain how u did it. just give the output and nothing else.
        """,
        expected_output=""" 
            Refined text containing only the information relevant to the question.
        """,
        agent=text_search_agent,
        tools=[search_tool],
    )

    # Crew Definition
    crew = Crew(
        agents=[text_search_agent],
        tasks=[text_retrieval_task],
        verbose=False,
    )

    # Execute the Task
    result = crew.kickoff(inputs={"question": question})
    return str(result)


def fetch_youtube_video_data(topic):
    """
    Function to search for YouTube videos matching a topic and summarize relevant video data.
    Returns a list of summarized video data including title, views, likes, comments, and channel info.
    
    Args:
        topic (str): The search topic for YouTube videos.
    
    Returns:
        list: A list of dictionaries with summarized YouTube video data.
    """
    # Load environment variables

    # YouTube API key from environment variables
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    # Build the YouTube API client
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # Set max_results to 5 inside the function
    max_results = 3

    try:
        # Search for videos
        search_response = youtube.search().list(
            q=topic,
            part="id,snippet",
            maxResults=max_results,
            type="video"
        ).execute()

        result = []
        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            channel_id = item["snippet"]["channelId"]
            channel_name = item["snippet"]["channelTitle"]

            # Fetch video statistics
            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            # Fetch channel statistics
            channel_response = youtube.channels().list(
                part="statistics",
                id=channel_id
            ).execute()

            if video_response["items"] and channel_response["items"]:
                stats = video_response["items"][0]["statistics"]
                channel_stats = channel_response["items"][0]["statistics"]

                views = stats.get("viewCount", "0")
                likes = stats.get("likeCount", "0")
                comments = stats.get("commentCount", "0")
                subscribers = channel_stats.get("subscriberCount", "0")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                result.append({
                    "title": title,
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "url": video_url,
                    "channel_name": channel_name,
                    "subscribers": subscribers
                })
            
        formatted_string = f"""
                            Title: {title}
                            Views: {views}
                            Likes: {likes}
                            Comments: {comments}
                            URL: {video_url}
                            Channel Name: {channel_name}
                            Subscribers: {subscribers}
                            """

        return str(formatted_string)

    except Exception as e:
        return f"An error occurred: {str(e)}"
    

if __name__ == "__main__":
    # Example usage
    print(fetch_youtube_video_data("programming hello world in assembly"))
    print(fetch_internet_search_results("samsung galaxy s25 expected release date"))
    












