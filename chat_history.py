import os
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set the environment variable.")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep your answers concise and accurate."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

class MessageHistorySession:
    def __init__(self):
        self.history: List[BaseMessage] = []

    def get_messages(self):
        return self.history

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message))

with_message_history = chain.with_message_history(
    MessageHistorySession(),
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history"
)

print("You (type exit to quit):")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = with_message_history.invoke({"input": user_input})
        
        print(f"AI: {response.content}")
        
    except Exception as e:
        print(f"Error: {e}")