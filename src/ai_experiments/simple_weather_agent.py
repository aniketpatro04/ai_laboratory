from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Defining a dummy weather tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

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
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", 
                   "content": "What's the weather in Bhubaneswar?"}]}
)


print(result["messages"][-1].content_blocks)