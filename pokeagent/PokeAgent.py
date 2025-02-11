import os
import requests
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import tool, render_text_description
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ============================================================
# Helper Functions and Tool Definitions
# ============================================================

# ---------------------------
# Existing Pokémon Info Tool
# ---------------------------
@tool
def fetch_pokemon_info(pokemon_name: str) -> dict:
    """Fetch and return all available raw information about a Pokémon using PokeAPI's 'pokemon' endpoint."""
    if not pokemon_name.strip():
        return {"error": "No Pokémon name provided."}
    
    # Build the URL using the provided Pokémon name.
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"Pokémon '{pokemon_name}' not found."}
        # Return the raw JSON data from the API.
        return response.json()
    except Exception as e:
        return {"error": f"An error occurred while fetching Pokémon data: {str(e)}"}

# --------------------------------------------------
# New Tool 1: Fetch High-Level Endpoints from PokeAPI
# --------------------------------------------------
@tool
def fetch_high_level_endpoints() -> dict:
    """
    Fetch and return the high-level endpoints available from the PokeAPI.
    For example, keys such as 'pokemon', 'ability', 'item', etc.
    """
    url = "https://pokeapi.co/api/v2/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to retrieve endpoints from PokeAPI (status code {response.status_code})."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# -----------------------------------------------------------
# New Tool 2: Dynamically Fetch Data from a Given Endpoint
# -----------------------------------------------------------
@tool
def fetch_from_endpoint(endpoint: str, query: str = "") -> dict:
    """
    Dynamically fetch data from a specified endpoint of the PokeAPI.
    
    Args:
        endpoint (str): The endpoint name (e.g., "pokemon", "ability").
        query (str): Optional query string or identifier to fetch a specific resource.
    
    Returns:
        dict: The JSON response from the API or an error message.
    """
    base_url = "https://pokeapi.co/api/v2"
    # Build the URL; if a query is provided, append it.
    if query:
        url = f"{base_url}/{endpoint}/{query.lower()}"
    else:
        url = f"{base_url}/{endpoint}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to retrieve data from '{url}' (status code {response.status_code})."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# ============================================================
# Define the Agent with a Custom Prompt Template
# ============================================================

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Create a list of tools (including both the existing and new ones)
tools = [
    fetch_pokemon_info,
    fetch_high_level_endpoints,
    fetch_from_endpoint
]

# Render text descriptions for the tools
tool_descriptions = render_text_description(tools)

# Define the prompt template (note the instructions now include the new tools)
template = """
Assistant is a language model that provides detailed information about Pokémon and can dynamically explore the PokeAPI.

**Instructions:**

1. **Use the available tools to retrieve information as needed.**

**Available Tools:**

{tools}

**Tool Names:**
{tool_names}

2. **Always follow the exact response format.**

3. **When answering, present the final answer in a clear and concise manner without including raw JSON.**

4. **If the question involves multiple API calls, use the appropriate tool for each call.**

**Response Format:**

- **If you need to use a tool:**

Thought: [Your thought process]  
Action: [tool name]  
Action Input: [input]

- **After receiving the observation, provide the Final Answer:**

Thought: [Your thought process]  
Final Answer: [Your answer to the user]

**Important:**

- Do not include both an Action and a Final Answer in the same message.
- Do not output raw JSON in your final answer.
- Only use the tools listed: {tool_names}

**Begin!**

New input: {input}
{agent_scratchpad}
"""

# Define input variables for the prompt template
input_variables = ["input", "agent_scratchpad"]

# Create the PromptTemplate
prompt_template = PromptTemplate(
    template=template,
    input_variables=input_variables,
    partial_variables={
        "tools": tool_descriptions,
        "tool_names": ", ".join([t.name for t in tools])
    }
)

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Create the AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=25
)

# ============================================================
# Streamlit App
# ============================================================

st.title("Pokémon Information Agent")
st.write("Ask questions about Pokémon and explore the PokeAPI using AI!")

user_input = st.text_input("Enter your Pokémon question:")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Send"):
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "agent_scratchpad": ""
            })
            if response is not None:
                final_answer = response.get('output', 'No answer provided.')
            else:
                final_answer = 'No answer provided.'
            
            st.write(f"**Question:** {user_input}")
            st.write(f"**Answer:** {final_answer}")
            st.session_state.conversation.append({"role": "assistant", "content": final_answer})
            st.session_state.chat_history = "\n".join(
                [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
            )
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            st.session_state.conversation.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

if st.session_state.conversation:
    st.write("## Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
