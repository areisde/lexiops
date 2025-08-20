from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse
import mlflow
from langfuse.langchain import CallbackHandler
from .tools.rag.server import rag_search, rag_citations
from .tools.tool_node import ToolNode
from dotenv import load_dotenv
import logging
import asyncio
import random
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)
langfuse_handler = CallbackHandler()
mlflow.langchain.autolog()


# Initialize the chat model
llm = init_chat_model("openai:gpt-4o")
tools = [rag_search, rag_citations]

# Define the graph state
class State(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Bind tools to the graph
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)

# Add nodes to the graph
def chatbot(state : State):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Setup tool routing
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Add conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile graph
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Set up talking function
def stream_graph_updates(task: str, config = None):
    async def async_stream():
        try:
            graph_ret = None
            if config is not None:
                graph_ret =  graph.astream({"messages": [{"role": "user", "content": task}]}, config, stream_mode="values")
            else:
                config_unique = {"configurable" : {"thread_id" : str(random.randint(1, 10000))},
                    "callbacks": [langfuse_handler]} # To be updated based on required thread
                graph_ret = graph.astream({"messages": [{"role": "user", "content": task}]}, config_unique, stream_mode="values")
            async for event in graph_ret:
                for value in event.values():
                    if isinstance(value, list):
                        for message in value:
                            logging.debug(f"Processing message: {message}")
                            if hasattr(message, "pretty_print"):
                                message.pretty_print()
                            else:
                                logging.warning(f"Message does not have pretty_print: {message}")
                    else:
                        logging.error(f"Unexpected value structure: {value}")
        except Exception as e:
            logging.error(f"Error in async_stream: {e}")

    asyncio.run(async_stream())


if __name__ == "__main__":
    config = {"configurable" : {"thread_id" : str(random.randint(1, 10000))},
              "callbacks": [langfuse_handler]} # To be updated based on required thread
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, config=config)
        except Exception as e:
            print("An error occurred:", e)
            break