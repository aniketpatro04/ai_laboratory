from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import requests

load_dotenv()

# Function to get coordinates for a city
def get_coordinates(city: str) -> tuple[float, float]:
    """Look up latitude/longitude for a city name using Open-Meteo's geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results")
    if not results:
        raise ValueError(f"Could not find coordinates for '{city}'")
    return results[0]["latitude"], results[0]["longitude"]

# Function to get weather for a city using Open-Meteo's weather API
def get_weather_from_openmeteo(city: str, latitude: float, longitude: float) -> str:
    """Get current weather and today's forecast for a given city using Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,precipitation_probability_max,apparent_temperature_max",
        "current": "temperature_2m",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current_temp = data["current"]["temperature_2m"]
        current_unit = data["current_units"]["temperature_2m"]

        today_max = data["daily"]["temperature_2m_max"][0]
        today_feels_like = data["daily"]["apparent_temperature_max"][0]
        rain_chance = data["daily"]["precipitation_probability_max"][0]

        return (
            f"Weather in {city}: currently {current_temp}{current_unit}. "
            f"Today's high: {today_max}{current_unit} (feels like {today_feels_like}{current_unit}), "
            f"with a {rain_chance}% chance of precipitation."
        )

    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't fetch the weather for {city} right now. ({e})"
    except (KeyError, IndexError):
        return f"Sorry, the weather data for {city} came back in an unexpected format."


# weather Tool Definition for agent
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    try:
        lat, lon = get_coordinates(city)
    except (requests.exceptions.RequestException, ValueError) as e:
        return f"Sorry, I couldn't locate '{city}'. ({e})"
    return get_weather_from_openmeteo(city, lat, lon)


# Intitialising the model
model = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
    max_tokens=1024,
    max_retries=2,
)

#Intializing the agent with the weather tool
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant. You can provide current weather information and today's forecast for any city in the world. Parse the city name from the user's query and use the Open-Meteo API to fetch the weather data. If the city is not found or if there is an error fetching the data, respond with an appropriate error message.",
)

result = agent.invoke(
    {"messages": [{"role": "user", 
                   "content": "What's the weather in Bhubaneswar?"}]}
)


print(result["messages"][-1].content_blocks)
