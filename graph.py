import uuid
from typing import Dict, Tuple, Any
import dotenv

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


def create_graph(
    retriever,
    model_name="gpt-3.5-turbo",
    system_prompt="",
    temp=None,
    api_key=None,
):
    def retrieve(state: MessagesState):
        print("---RETRIEVE---")
        # print(state)
        question = state["messages"][-1].content
        documents = retriever.invoke(question, top_k=3)
        # print("question:", question)
        # print("documents:", documents)

        response = [
            system_prompt,
            "Context: ",
            "\n\n".join(doc.page_content for doc in documents),
            "Question: " + question,
            "Answer:",
        ]
        response_str = "\n".join(response)

        # print("RESPONSE:", response)
        state["messages"] += [HumanMessage(content=response_str)]
        return {"messages": state["messages"]}

    def generate(state: MessagesState):
        print("---GENERATE---")
        # print(state)
        response = llm.invoke(state["messages"])
        return {"messages": response}

    llm = ChatOpenAI(model=model_name, temperature=temp, api_key=api_key)
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
    print(response["messages"][-1].pretty_print())
    response = graph.invoke(
        {"messages": [HumanMessage(content="다른 떡볶이 맛집 추천해줘.")]},
        config=chatConfig,
    )
    print(response["messages"][-1].pretty_print())
