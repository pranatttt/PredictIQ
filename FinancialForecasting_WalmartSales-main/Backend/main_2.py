from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Initialize FastAPI app
app = FastAPI()

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

# === OLLAMA NATURAL LANGUAGE CHAT MODEL ===
llm_natural = ChatOllama(
    model="llama3.2",             # ✅ updated for your system
    base_url="http://localhost:11434",
)

# Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful sales assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Conversation memory + chain
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

conversation_chain = ConversationChain(
    llm=llm_natural,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

# === /chat NATURAL LANGUAGE ENDPOINT ===
@app.post("/chat")
async def chat(input: ChatInput):
    response = await run_in_threadpool(conversation_chain.run, input.message)
    return {"response": response, "source": "llm_natural"}
