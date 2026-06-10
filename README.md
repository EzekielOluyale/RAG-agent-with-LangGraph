## RAG-agent-with-LangChain
This is a chatbot built to answer questions about my resume and professional background.

## 🧭 How It Works (Step-by-Step)
```text
  [ User Asks a Question ]
             │
             ▼
   1. Understand the Question (Should I search the resume?)
         /          \
      (Yes)         (No) ──► [ Politely decline and stop ]
       /
      ▼
   2. Search the Resume (Look inside the Pinecone database)
             │
             ▼
   3. Check the Quality (Did we find the right information?)
         /          \
      (Yes)         (No)
       /              \
      ▼                ▼
4. Give Final Answer   5. Rewrite the Question (Fix search term & loop back to Step 1)
```

- Understand the Question: The chatbot checks if your question is about Ezekiel. If you ask about something random (like a cooking recipe), it will politely say no.

- Search the Resume: It searches a Pinecone database where Ezekiel's resume is stored.

- Check the Quality: The AI reads what it found and asks itself: "Does this text actually answer the user's question?"

- Rewrite the Question: If the database returned the wrong information, the chatbot doesn't give up. It rephrases the question to make it cleaner and loops back to try the search again.

- Give Final Answer: Once it finds the perfect information, it writes a short, professional response (maximum of 3 sentences).

## Tech/Production Stack:
- LangGraph: Orchestrates the agent workflow - handles multi-step logic
- Supabase: Managing conversational state - short term memory
- Pinecone: Knowledgeable AI - long term memory
- FastAPI: Backend - handling asynchronous network requests
- Render: Server - runs the Docker container that packages the FastAPI backend
- Streamlit: Frontend

## How to Run It Locally
Clone this project:
```
git clone [https://github.com/EzekielOluyale/RAG-agent-with-LangGraph.git](https://github.com/EzekielOluyale/RAG-agent-with-LangGraph.git)
cd RAG-agent-with-LangGraph
```
Add your secret keys: Create a file named .env in the root folder and add your keys
```
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
DATABASE_URL=your_supabase_url
BACKEND_URL=your_fastapi_url
```

## Installing
To install all required dependencies:
```sh
pip install -r requirements.txt
```

## IMPORTANT STEP
Before starting the chatbot, you must upload Ezekiel's resume data into Pinecone.
- Open and run all the cells inside experiment.ipynb.
- This will read the resume text, turn it into vectors, and save it safely in your Pinecone index.

## Run the Streamlit Web App
```
streamlit run main.py
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📂 Folder Structure
```
Rag-agent-with-LangGraph/
├── data
│   ├── ...
├── notebooks
│   ├── ...              
├── src
│   ├── ...         
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENCE
├── README.md
├── docker-compose.yml
├── main.py
├── requirements.txt     
├── setup.py
├── streamlit.py     
```

## ❓ Help
If you encounter any issues:
- Open an issue in this repository

## ✍️ Author
👤 Oluyale Ezekiel
- 📧 Email: ezekieloluyale@gmail.com
- LinkedIn: [Ezekiel Oluyale](https://www.linkedin.com/in/ezekiel-oluyale)
