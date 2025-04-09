from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

result = llm.invoke("Who is the prime minister of UK?")

# print(result)
print(result.content)