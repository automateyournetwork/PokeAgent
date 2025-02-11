PokeAgent
🚀 A ReAct AI Agent for Pokémon Powered by PokeAPI & OpenAI

PokeAgent is an AI-driven assistant that retrieves Pokémon data dynamically using PokeAPI. Built with LangChain, OpenAI, and Streamlit, it leverages a ReAct (Reasoning + Acting) agent to provide intelligent responses about Pokémon abilities, types, stats, and more.

🛠 Installation Options
1️⃣ Option 1: Run Locally (Virtual Environment)
Step 1: Clone the Repository
```bash
git clone https://github.com/automateyournetwork/PokeAgent.git
cd PokeAgent
```

Alternatively, download the repository as a ZIP file and extract it.

Step 2: Set Up a Virtual Environment
Using Python's built-in virtual environment:

``` bash
python -m venv venv
```

Activate the virtual environment:

🔹 Windows:

```bash
venv\Scripts\activate
```

🔹 macOS/Linux:

```bash
source venv/bin/activate
```

Step 3: Install Dependencies

``` bash
pip install -r requirements.txt
```

Dependencies (requirements.txt):

streamlit – Interactive UI
requests – API requests
langchain – AI agent framework

Step 4: Set Up OpenAI API Key

Create a .env file inside the PokeAgent folder and add your OpenAI API key:

```bash
OPENAI_API_KEY="<your_api_key>"
```

Step 5: Run the Streamlit App

```bash
streamlit run PokeAgent.py
```

Visit localhost:8501 to access the app.

2️⃣ Option 2: Run with Docker
Use Docker Compose to containerize the application for easy setup.

Step 1: Clone the Repository
```bash
git clone https://github.com/automateyournetwork/PokeAgent.git
cd PokeAgent
```

Step 2: Create a .env File

Inside the PokeAgent folder, create a file named .env and add your OpenAI API key:

```bash
OPENAI_API_KEY="<your_api_key>"
```

Step 3: Run with Docker Compose

Ensure you have Docker Desktop or Docker Compose installed. Then, run:

```bash
docker-compose up --build
```

This will:
✅ Pull necessary dependencies
✅ Start the Streamlit frontend
✅ Expose the app on localhost:8501

To stop the container, press CTRL + C or run:

``` bash
docker-compose down
``` 

🎮 Start Asking Questions!
Once the app is running, ask questions like:

"Tell me about Pikachu!"
"What are Bulbasaur's abilities?"
"Compare Charizard and Blastoise."

PokeAgent will fetch live data from PokeAPI and respond intelligently! 🚀

📌 Troubleshooting
🔹 OpenAI API Key Not Found?

Ensure .env exists inside the PokeAgent folder.
Restart the app after setting the key.

🔹 Streamlit App Not Running?

Check if port 8501 is free.
Try running:

``` bash

streamlit run PokeAgent.py --server.port 8502
``` 

🔹 Docker Issues?

Ensure Docker Desktop is running.

Run docker ps to check if the container is active.
⭐ Contribute & Improve

Feel free to submit issues, PRs, or suggestions via GitHub.

🚀 Happy Pokémon Researching! Gotta AI 'Em All! 🎉
