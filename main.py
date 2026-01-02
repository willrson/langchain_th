import os
import requests
from dotenv import load_dotenv
from langsmith import Client
from openai import OpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# Define the state for our graph
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def get_weather(city: str) -> str:
    """Get real weather for a given city using wttr.in"""
    try:
        url = f"http://wttr.in/{city}?format=j1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract weather information
        current = data['current_condition'][0]
        weather_desc = current['weatherDesc'][0]['value']
        temp_c = current['temp_C']
        feels_like_c = current['FeelsLikeC']
        humidity = current['humidity']
        
        # Get location info
        location = data['nearest_area'][0]
        city_name = location['areaName'][0]['value']
        country = location['country'][0]['value']
        
        return f"Weather in {city_name}, {country}: {weather_desc}. Temperature: {temp_c}°C (feels like {feels_like_c}°C). Humidity: {humidity}%."
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing weather data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_temperature_records(city: str) -> str:
    """Get historical highest and lowest temperature records for a city from the self-made LangSmith dataset, with OpenAI fallback."""
    try:
        # Initialize LangSmith client
        client = Client()
        
        # Use your self-made dataset
        dataset_name = "self-made-temperatures"
        
        # Check if dataset exists
        if not client.has_dataset(dataset_name=dataset_name):
            return f"Temperature records dataset '{dataset_name}' not found. Please check the dataset name."
        
        # Get all examples (small dataset with only 11 entries)
        examples = list(client.list_examples(dataset_name=dataset_name))
        
        # Look for the city in the dataset
        city_lower = city.lower()
        
        for example in examples:
            if hasattr(example, 'inputs') and example.inputs:
                # Dataset structure: inputs={'city': 'CityName'}, outputs={'temp_max': X, 'temp_min': Y}
                dataset_city = example.inputs.get('city', '')
                if isinstance(dataset_city, str) and dataset_city.lower() == city_lower:
                    # Found matching city, extract temperature data
                    if hasattr(example, 'outputs') and example.outputs:
                        temp_max = example.outputs.get('temp_max', 'N/A')
                        temp_min = example.outputs.get('temp_min', 'N/A')
                        return f"Temperature records for {dataset_city} (from dataset): Highest recorded: {temp_max}°C, Lowest recorded: {temp_min}°C"
                    else:
                        return f"Found data for '{dataset_city}' but temperature information is not available."
        
        # City not found in dataset - fallback to OpenAI
        return _query_openai_for_temperature_records(city)
            
    except Exception as e:
        return f"Error accessing temperature records: {str(e)}"

def _query_openai_for_temperature_records(city: str) -> str:
    """Fallback function to query OpenAI for temperature records when city is not in dataset."""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Please provide the historical highest and lowest temperature records for {city}. 
        Format your response as: "Temperature records for [City]: Highest recorded: [X]°C, Lowest recorded: [Y]°C"
        If you don't have exact data, provide reasonable estimates based on the city's climate and mention they are estimates.
        Keep the response concise and factual."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a weather data expert. Provide accurate historical temperature records or reasonable estimates."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1  # Low temperature for more factual responses
        )
        
        result = response.choices[0].message.content.strip()
        return f"{result} (from OpenAI - not in local dataset)"
        
    except Exception as e:
        # List available cities from dataset as final fallback
        try:
            client = Client()
            examples = list(client.list_examples(dataset_name="self-made-temperatures"))
            available_cities = []
            for example in examples:
                if hasattr(example, 'inputs') and example.inputs:
                    city_name = example.inputs.get('city', '')
                    if city_name:
                        available_cities.append(city_name)
            
            return f"No temperature records found for '{city}' in dataset and OpenAI query failed ({str(e)}). Available cities in dataset: {', '.join(available_cities)}"
        except:
            return f"Error querying both dataset and OpenAI for temperature records: {str(e)}"

# Create the LangGraph workflow
def create_weather_agent():
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to the model
    tools = [get_weather, get_temperature_records]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define the chatbot node
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Create the graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[get_weather, get_temperature_records])
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder.compile()

# Create the agent
agent = create_weather_agent()

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the temperature records for Barcelona?"}]}
    )
    
    print("Agent Response:")
    print(result["messages"][-1].content)