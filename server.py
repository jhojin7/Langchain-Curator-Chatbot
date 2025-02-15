from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Any
import streamlit as st
from csv_retrieve import rag_from_csv, create_retriever
from pathlib import Path
from graph import create_graph
from st_main_ui import generate_response, strip_doc
import os


system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""
retriever = create_retriever(
    csv_path=Path("cache", "all_reviews_sampled.csv"),
    top_k=3,
    save_path=Path("faiss_cache_dir", "all_reviews_sampled.csv"),
)
openai_model_name = "gpt-4o-mini"
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
graph, config = create_graph(
    retriever=retriever,
    model_name=openai_model_name,
    system_prompt=system_prompt,
    temp=0.1,
    api_key=openai_api_key,
)


app = FastAPI()


class ChatRequest(BaseModel):
    query: str = Field(title="User query", examples=["떡볶이  맛집 추천해줘."])
    messages: List[Any] = Field(
        title="Chat messages",
        examples=[{"role": "user", "content": "떡볶이 맛집 추천해줘."}],
    )


class ChatResponse(BaseModel):
    docs: List[Any]
    response: str


from graph import create_graph
from csv_retrieve import create_retriever


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response_msg, retrieved_docs = generate_response(
        query=request.query, graph=graph, messages=request.messages
    )
    docs = [strip_doc(doc) for doc in retrieved_docs]
    chat_respose = None
    if not docs or not response_msg:
        response_msg = "I don't know the answer."
        chat_respose = ChatResponse(docs=[], response=response_msg)
    chat_respose = ChatResponse(docs=retrieved_docs, response=response_msg)
    print(chat_respose)
    return chat_respose


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8081, reload=True)
