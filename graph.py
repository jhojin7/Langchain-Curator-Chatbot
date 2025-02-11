import uuid
from typing import Dict, Tuple, Any
import dotenv

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import streamlit as st


def create_graph(
    retriever,
    model_name="gpt-3.5-turbo",
    system_prompt="",
    temp=None,
    api_key=None,
) -> Tuple[StateGraph, Dict[str, Any]]:
    llm = ChatOpenAI(model=model_name, temperature=temp, api_key=api_key)

    def retrieve(state: MessagesState):
        question = state["messages"][-1].content
        documents = retriever.invoke(question)

        return {
            "messages": [ToolMessage(content=list(documents), tool_call_id="retriever")]
        }

    def generate(state: MessagesState):
        question = state["messages"][-2].content
        retrieved_docs = state["messages"][-1].content
        context = "\n\n".join(retrieved_docs)
        to_combine = [
            system_prompt,
            "Context: ",
            context,
            "Question: " + question,
            "Answer:",
        ]
        combined_text = "\n".join(to_combine)
        response = llm.invoke(combined_text)
        return {"messages": response}

    graph_builder = StateGraph(state_schema=MessagesState)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    graph = graph_builder.compile(checkpointer=MemorySaver())
    chatConfig = {"configurable": {"thread_id": uuid.uuid4()}}
    return graph, chatConfig


if __name__ == "__main__":
    from csv_retrieve import create_retriever
    from pathlib import Path

    retriever = create_retriever(Path("cache/네이버맛집리스트_20250201.0105.csv"))
    print(retriever)
    graph, chatConfig = create_graph(retriever)
    print(graph)
    response = graph.invoke(
        {"messages": [HumanMessage(content="떡볶이 맛집 추천해줘.")]}, config=chatConfig
    )
    messages = graph.get_state(config=chatConfig).values["messages"]
    print(messages)
