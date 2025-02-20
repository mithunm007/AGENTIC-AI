import sys
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from MarkdownTools import MarkdownValidationTool  # Import your custom tool
import os

load_dotenv()

# Set up the LLM (Ollama or ChatOpenAI)
# ollama_llm = ChatOllama(
#     model="mistral:latest",
#     base_url="http://localhost:11434"
# )
llm = ChatOpenAI(
    model="ollama/mistral",
    base_url="http://localhost:11434"
)

# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="groq/mixtral-8x7b-32768"
# )

# Define the MarkdownValidationTool
markdown_validation_tool = MarkdownValidationTool()

# Define the Agent and pass the custom tool
markdown_linting_agent = Agent(
    role='Requirements Manager',
    goal="""Provide a detailed list of the markdown 
            linting results. Give a summary with actionable 
            tasks to address the validation results. Write your 
            response as if you were handing it to a developer 
            to fix the issues.
            DO NOT provide examples of how to fix the issues or
            recommend other tools to use.""",
    backstory="""You are an expert business analyst 
            and software QA specialist. You provide high quality, 
            thorough, insightful and actionable feedback via 
            detailed list of changes and actionable tasks.""",
    verbose=True,
    llm=llm
)

# Define the task, specifying the custom tool
syntax_review_task = Task(
    description="""Use the MarkdownValidationTool to review the file(s)
        Pass only the file path to the MarkdownValidationTool. Use the following format to call the tool:
        Do I need to use a tool? Yes
        Action: markdown_validation_tool
        Action Input: {filename}

        Get the validation results from the tool and then summarize them into a list of changes
        the developer should make to the document. DO NOT recommend ways to update the document.
        DO NOT change any of the content of the document or add content to it.
        
        If you already know the answer or if you do not need to use a tool, return it as your Final Answer.""",
    agent=markdown_linting_agent,
    expected_output="""Return all the markdown validation issues with actionable feedback.""",
    tools=[markdown_validation_tool]  # Use the custom tool
)

# Define the Crew with the agent and tasks
crew = Crew(
    agents=[markdown_linting_agent],
    tasks=[syntax_review_task],
    verbose=True,
)

inputs = {"filename":sys.argv[1]}
result = crew.kickoff(inputs=inputs)
print(result)
