from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    SystemMessage("You are an expert in cricket."),
    HumanMessage("Who is called badshah of bollywood?")
]

result = llm.invoke(messages)

# print(result)
print(result.content)