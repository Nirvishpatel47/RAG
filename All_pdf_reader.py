import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

# --- Environment and API Key Setup ---
try:
    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

except Exception as e:
    print(f"Error: {e}")

# --- Document Loading and Splitting ---
pdf_path = input("Enter PAth of your PDF: ")

try:
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
except NotImplemented:
    print("No such PDF.")
except Exception as e:
    print(f"Error {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(document)

print("Text split complete.")

messages = [
    AIMessage(content="Hello! How can i help you today?")
]
# --- Embedding and Vector Store ---
print("Creating embeddings and vector store...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
vectorizer = FAISS.from_documents(documents=docs, embedding=embeddings)
vectorizer.save_local("Vec")
print("Vector store saved to 'Vec'.")

# --- Language Model and RAG Chain Setup ---
print("Setting up the RAG chain...")
retriever = vectorizer.as_retriever(search_type="similarity", kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=200, google_api_key=gemini_api_key)

past_history_ = ChatPromptTemplate.from_template(
    """You have only task to generate a very good summary of following text from the above text with max_tokens or max_words 200. 
    Make sure that you priotize the last chat.
    The below text is the text you have to summarize:
    {messages}"""
)
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know rather than hallucinating.
    \n\n
    This is the past history of messages also try to answer question based on past summary: {past_history}\n\n
    {context}\n\n
    Question: {question}"""
)

parser = StrOutputParser()

chain = past_history_ | llm | parser
# New, robust RAG chain definition
rag_chain = (
    RunnableParallel(
        past_history = chain,
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | parser
)

print("RAG chain setup complete.")

# --- Invoking the Chain ---
while True:
    question_to_ask = input("Enter your question as per PDF or exit: ")
    messages.append(HumanMessage(content=question_to_ask))
    if question_to_ask.lower() == "exit":
        print("Thanks for using this app.")
        break

    print(f"Invoking chain with question: '{question_to_ask}'")

    response = rag_chain.invoke(question_to_ask)
    messages.append(response)
    print("\n--- Response ---")
    print(response)