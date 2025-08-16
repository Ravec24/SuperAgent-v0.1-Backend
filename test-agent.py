# test_agent.py
from agent import get_agent
import os
from dotenv import load_dotenv

load_dotenv()

def test_agent():
    agent = get_agent()
    
    test_queries = [
        "Hello, what can you help me with?",
        "What's the weather like in Tokyo?",
        "Calculate 25 * 17 + 33",
        "Search for information about LangGraph"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        try:
            result = agent.invoke({
                "messages": [("human", query)]
            })
            response = result["messages"][-1].content
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_agent()
