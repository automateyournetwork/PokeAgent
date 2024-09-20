import os
import json
import requests
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.tools import tool, render_text_description
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Load the Pokémon data from pokemon.json
def load_pokemon_data(file_path: str = 'pokemon.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File '{file_path}' not found."}
    except json.JSONDecodeError:
        return {"error": f"Error decoding JSON from file '{file_path}'."}

# Find the URL of a Pokémon in the JSON file
def find_pokemon_url(pokemon_name: str, pokemon_data: list):
    pokemon_name = pokemon_name.lower()
    for entry in pokemon_data:
        if entry['name'].lower() == pokemon_name:
            return entry['url']
    return None

# Fetch Pokémon information from the API using the URL
def get_pokemon_info_from_url(pokemon_url: str):
    try:
        response = requests.get(pokemon_url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Unable to fetch Pokémon data from '{pokemon_url}'."}
    except Exception as e:
        return {"error": f"An error occurred while fetching Pokémon data: {str(e)}"}

# Tool to fetch and extract Pokémon information
@tool
def fetch_pokemon_info(pokemon_name: str) -> dict:
    """Fetch and return key information about a Pokémon."""
    # Load the Pokémon data
    pokemon_data = load_pokemon_data()

    if "error" in pokemon_data:
        return pokemon_data

    # Find the Pokémon URL
    pokemon_url = find_pokemon_url(pokemon_name, pokemon_data)
    if not pokemon_url:
        return {"error": f"Pokémon '{pokemon_name}' not found in the local database."}

    # Fetch the Pokémon data
    full_data = get_pokemon_info_from_url(pokemon_url)

    if "error" in full_data:
        return full_data

    # Extract relevant information
    extracted_info = {
        "name": full_data.get("name", "N/A").capitalize(),
        "id": full_data.get("id", "N/A"),
        "height": full_data.get("height", "N/A"),
        "weight": full_data.get("weight", "N/A"),
        "types": [t["type"]["name"] for t in full_data.get("types", [])],
        "abilities": [a["ability"]["name"] for a in full_data.get("abilities", [])],
        "stats": {s["stat"]["name"]: s["base_stat"] for s in full_data.get("stats", [])},
        "sprite": full_data.get("sprites", {}).get("front_default", None),
    }

    return extracted_info

# ============================================================
# Define the agent with a custom prompt template
# ============================================================

# Initialize the Ollama LLM
llm = Ollama(model="llama3.1")

# Create a list of tools
tools = [fetch_pokemon_info]

# Render text descriptions for the tools
tool_descriptions = render_text_description(tools)

template = """
Assistant is a language model that provides detailed information about Pokémon.

**Instructions:**

1. **Use the 'fetch_pokemon_info' tool to retrieve Pokémon information when needed.**

2. **Always follow the exact response format to avoid errors.**

3. **Present the final answer in a clear and concise manner, summarizing key information without including raw JSON data.**

**Response Format:**

When using a tool:

Thought: Do I need to use a tool? Yes Action: [tool name] Action Input: [input] Observation: [result]

arduino
Copy code

After receiving the observation, provide the final answer:

Thought: [Your thought process] Final Answer: [Your answer to the user]

sql
Copy code

**Important:** Do not include raw JSON in the final answer. Extract relevant information and present it neatly.

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""

# Define input variables
input_variables = ["input", "agent_scratchpad", "chat_history"]

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
agent = create_react_agent(llm, tools, prompt_template)

# ============================================================
# Streamlit App
# ============================================================

# Initialize the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5
)

# Initialize Streamlit
st.title("Pokémon Information Agent")
st.write("Ask questions about Pokémon and get information using AI!")

# Input for user questions
user_input = st.text_input("Enter your Pokémon question:")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Button to submit the question
if st.button("Send"):
    if user_input:
        # Add the user input to the conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})

        # Invoke the agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": ""
        })

        # Extract the final answer
        final_answer = response.get('output', 'No answer provided.')

        # Display the question and answer
        st.write(f"**Question:** {user_input}")
        st.write(f"**Answer:** {final_answer}")

        # Add the response to the conversation history
        st.session_state.conversation.append({"role": "assistant", "content": final_answer})

        # Update chat history
        st.session_state.chat_history = "\n".join(
            [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
        )

# Display the conversation history
if st.session_state.conversation:
    st.write("## Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")