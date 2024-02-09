import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()


def getVectorstoreFromURL(url):
    # Get the webpage in documents form
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split the text into sentences
    textSplitter = RecursiveCharacterTextSplitter()
    documentChuns = textSplitter.split_documents(documents)

    # Get the vectorstore
    vectorstore = Chroma.from_documents(documentChuns, OpenAIEmbeddings())

    return vectorstore


def getContextRetrieveChain(vectorStore):
    llm = ChatOpenAI()

    retriever = vectorStore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chatHistory"),
        ("user", "{input} "),
        ("user", "Given the above conversation, generat a search query to look up in order to get information about the topic.")
    ])

    retrieverChain = create_history_aware_retriever(llm, retriever, prompt)

    return retrieverChain


def getConversationalRagChain(retrieverChain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chatHistory"),
        ("user", "{input} ")
    ])

    stuffDocumentsChain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retrieverChain, stuffDocumentsChain)


def getResponse(user_input):
    retrieverChain = getContextRetrieveChain(st.session_state.vectorStore)

    conversationalRagChain = getConversationalRagChain(retrieverChain)

    response = conversationalRagChain.invoke({
        "chatHistory": st.session_state.chatHistory,
        "input": userQuery
    })

    return response['answer']

# App config
st.set_page_config(page_title="Chat with website", page_icon="robot")
st.title("Chat with website")

# Sidebar
with st.sidebar:
    st.write("This is a chat with website. You can chat with the website by typing in the input box below.")
    st.header("Settings")
    websiteURL = st.text_input("Website URL")
    st.write("You can chat with the website by typing in the input box below.")


if websiteURL is None or websiteURL == "":
    st.info("Please enter a website URL")
else:

    # Store previous conversation
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # Create conversation chain
    if "vectorStore" not in st.session_state:
        st.session_state.vectorStore = getVectorstoreFromURL(websiteURL)

   
    # User input
    userQuery = st.chat_input("Type your message here...")

    if userQuery is not None and userQuery != "": 
        response = getResponse(userQuery) 

        st.session_state.chatHistory.append(HumanMessage(content=userQuery))
        st.session_state.chatHistory.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chatHistory:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
