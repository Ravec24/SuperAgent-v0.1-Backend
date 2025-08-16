# src/agent.py
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Define custom tools for your agent

@tool
def get_weather(city: str) -> str:
    """
    Get real-time weather information for a city using WeatherAPI.
    Returns temperature and weather condition.
    """
    import os, requests
    
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        return "Weather API key not found. Please set WEATHER_API_KEY in your .env file."

    try:
        url = "http://api.weatherapi.com/v1/current.json"
        resp = requests.get(url, params={"key": API_KEY, "q": city})
        data = resp.json()

        if "error" in data:
            return f"Error: {data['error']['message']}"

        temp = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"The temperature in {city} is {temp}°C with {condition.lower()}."
    
    except Exception as e:
        return f"Could not fetch weather for {city}: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    """
    try:
        # Basic calculator - extend as needed
        # Note: Use ast.literal_eval for better security in production
        result = eval(expression)
        return f"The result is: {result}"
    except:
        return "Invalid expression"


@tool
def story_teller(prompt: str) -> str:
    """
    Generate a short, engaging story based on the input prompt.
    """
    # Rely on the LLM to generate the story via prompting
    return f"Tell a short story about: {prompt}"


@tool
def sentence_rewriter(sentence: str, tone: str = "clear") -> str:
    """
    Rewrite the given sentence in the specified tone ('clear', 'friendly', 'professional', etc).
    """
    return f"Rewrite this sentence in a {tone} way: {sentence}"

@tool
def search_web(query: str) -> str:
    """
    Perform a web search using Tavily API and return the first relevant snippet.
    """
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "search_depth": "basic", "include_answer": True}
        )
        result = resp.json()
        if result.get("results") and len(result["results"]) > 0:
            snippet = result["results"][0].get("content", "No snippet available.")
            url = result["results"][0].get("url", "")
            return f"{snippet}\nSource: {url}"
        else:
            return "No result found."
    except Exception as e:
        return f"Could not search the web: {str(e)}"



# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# Create the prebuilt ReAct agent
agent = create_react_agent(
    llm, 
    tools=[get_weather, calculate, search_web, story_teller, sentence_rewriter],
    prompt="You are a helpful AI assistant powered by Google Gemini. Use the available tools to answer user questions accurately and helpfully."
)


# Export the agent
def get_agent():
    return agent
