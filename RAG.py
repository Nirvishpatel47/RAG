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

# --- Environment and API Key Setup ---
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# --- Document Loading and Splitting ---
pdf_path = r"D:\Projects\Why We SLEEP.pdf"
loader = PyPDFLoader(pdf_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(document)

print("Text split complete.")

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

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know rather than hallucinating.
    \n\n
    {context}\n\n
    Question: {question}"""
)

parser = StrOutputParser()

# New, robust RAG chain definition
rag_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | parser
)

print("RAG chain setup complete.")

# --- Invoking the Chain ---
question_to_ask = "How to sleep well?"
print(f"Invoking chain with question: '{question_to_ask}'")

response = rag_chain.invoke(question_to_ask)

print("\n--- Response ---")
print(response)