import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

def get_weather(city: str) -> str:
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

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant that provides real-time weather information for cities around the world.",
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in London"}]}
    )
    
    print("Agent Response:")
    print(result["messages"][-1].content)