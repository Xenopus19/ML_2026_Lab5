import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# 1. Заменили импорт Anthropic на Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from server import search_books, add_new_author, add_new_book, link_book_to_author


# --- Convert async Database functions to LangChain Tools ---
@tool
def tool_search_books(title_query: str) -> str:
    """Search for books in the library database by their title."""
    return asyncio.run(search_books(title_query))


@tool
def tool_add_new_author(name: str, nationality: str = None) -> str:
    """Adds a new author to the library database. Always use this before adding a book if the author is new."""
    return asyncio.run(add_new_author(name, nationality))


@tool
def tool_add_new_book(title: str, published_year: int = None) -> str:
    """Adds a new book to the library database and returns its ID."""
    return asyncio.run(add_new_book(title, published_year))


@tool
def tool_link_book_to_author(book_id: int, author_id: int) -> str:
    """Links an existing book to an existing author using their IDs."""
    return asyncio.run(link_book_to_author(book_id, author_id))


my_tools = [tool_search_books, tool_add_new_author, tool_add_new_book, tool_link_book_to_author]

st.title("📚 Local MCP Desktop")

# --- UI Sidebar: Model Selection ---
with st.sidebar:
    st.header("Settings")
    model_provider = st.selectbox(
        "Choose Model Provider",
        ("Local (Ollama)", "Google (Gemini)")
    )

    if model_provider == "Google (Gemini)":
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
    else:
        st.info("Ensure Ollama is running locally (e.g., `ollama run llama3.2`).")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about the library database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE AGENTIC ORCHESTRATION LOOP ---
    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        st_placeholder.markdown("🤔 Thinking...")

        try:
            # Step 1: Initialize LLM
            if model_provider == "Local (Ollama)":
                llm = ChatOllama(model="llama3.2", base_url="http://ollama-service:11434")
            else:
                if not api_key:
                    st.error("Please enter your API Key in the sidebar.")
                    st.stop()
                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, transport="rest")

            # Step 2: Bind the tools to the brain
            llm_with_tools = llm.bind_tools(my_tools)

            conversation_history = [HumanMessage(content=prompt)]

            # Step 3: Ask the LLM what it wants to do
            ai_msg = llm_with_tools.invoke(conversation_history)
            conversation_history.append(ai_msg)

            # Step 4: Check if the LLM decided to use a database tool!
            while ai_msg.tool_calls:
                st_placeholder.markdown("🛠️ Querying the database...")

                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "tool_search_books":
                        result = tool_search_books.invoke(tool_args)
                    elif tool_name == "tool_add_new_author":
                        result = tool_add_new_author.invoke(tool_args)
                    elif tool_name == "tool_add_new_book":
                        result = tool_add_new_book.invoke(tool_args)
                    elif tool_name == "tool_link_book_to_author":
                        result = tool_link_book_to_author.invoke(tool_args)
                    else:
                        result = "Error: Tool not found."

                    conversation_history.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

                # Step 5: Let the LLM use the results or request another tool
                st_placeholder.markdown("🧠 Reading database results...")
                ai_msg = llm_with_tools.invoke(conversation_history)
                conversation_history.append(ai_msg)

            response_text = ai_msg.text

            st_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st_placeholder.error(f"Error: {repr(e)}")