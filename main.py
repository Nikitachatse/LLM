import os
import boto3
import streamlit as st
from dotenv import load_dotenv
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory

custom_css = """
<style>
:root {
    --pc9: rgb(27, 27, 27);
    --bg-color: rgb(27, 27, 27);
}

[data-testid="stAppViewContainer"] {
    color: var(--pc9) !important;
    background: 
        linear-gradient(-90deg, rgba(170, 170, 170, .07) 1px, transparent 1px),
        linear-gradient(rgba(170, 170, 170, .07) 1px, transparent 1px),
        linear-gradient(-90deg, rgba(170, 170, 170, .03) 1px, transparent 1px),
        linear-gradient(rgba(170, 170, 170, .03) 1px, transparent 1px),
        linear-gradient(transparent 4px, var(--bg-color) 4px, var(--bg-color) 77px, transparent 77px),
        linear-gradient(-90deg, rgba(170, 170, 170, .05) 1px, transparent 1px),
        linear-gradient(-90deg, transparent 4px, var(--bg-color) 4px, var(--bg-color) 77px, transparent 77px),
        linear-gradient(rgba(170, 170, 170, .05) 1px, transparent 1px),
        var(--bg-color);
    background-size: 15px 15px, 15px 15px, 60px 60px, 60px 60px, 60px 60px, 60px 60px, 60px 60px, 60px 60px;
    font-family: truenorg !important;
    font-size: 13px;
    font-weight: 400;
    line-height: 1.471;
}
[data-testid="stAppViewContainer"] * {
    color: white !important;
}

[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea {
    color: black !important;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Load environment variables
load_dotenv("credentials.env")

# --- Bedrock Config ---
config = Config(retries={"max_attempts": 10, "mode": "standard"})

# --- Streamlit UI Title ---
st.title("Hello, Welcome To SQL-LLM")

# --- Initialize Bedrock + LangChain in Session State ---
if "conversation" not in st.session_state:
    # Boto3 client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("REGION_NAME"),
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        config=config
    )

    # LangChain ChatBedrock LLM
    llm = ChatBedrock(
        model_id=os.getenv("MODEL_ID"),
        client=client,
        model_kwargs={"temperature": 0, "max_tokens": 8192},
        streaming=True
    )

    # Memory setup
    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)

    with open("SQL_query.txt", "r") as f:
        file_content = f.read()
    SYSTEM_PROMPT = f"""
You are an AI assistant that helps analyze and explain ANSI SQL expressions organized in a DAG (Directed Acyclic Graph) structure.
The structure is made of:
- STAGES, numbered as STAGE 0, STAGE 1, etc.
- Each stage contains a Model, defined like: `STAGE X - Model Name : <Model Name>`
- Each model includes one or more TASKS, listed as `# TASK 0`, `# TASK 1`, etc.

Each task has the following information:
- `Task Desc`: A short description of what the task does.
- `Target Map name`: A name identifying the target mapping.
- `Target Table name`: The final table where results will be stored.
- `SQL`: An ANSI SQL query enclosed within triple quotes 

Your job is to:
1. Parse and understand the structure of stages, models, and tasks.
2. Identify each SQL block and the context it belongs to (which task and model).
3. Be able to answer any follow-up questions.

Read code file : \n{file_content}.
"""
    memory.chat_memory.add_user_message(SYSTEM_PROMPT)

    # Save conversation chain to session
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    st.session_state.chat_history_display = []  # for Streamlit rendering

# --- Render Previous Messages ---
for message in st.session_state.chat_history_display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Say something..."):
    # Show user message
    st.session_state.chat_history_display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response = st.session_state.conversation.run(prompt)
        st.markdown(response)

    # Save assistant message
    st.session_state.chat_history_display.append({"role": "assistant", "content": response})
