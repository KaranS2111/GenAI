from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

msg = [("system", "You have knowledge about {celebrity}."),
       ("human", "Tell me about this person")]

promt_msg = ChatPromptTemplate.from_messages(msg)

output = RunnableLambda(lambda y : y.content)

chain = promt_msg | llm | output

print("CELEB TALK!")
my_input = input("Write the name of your favourite celebrity : ")
result = chain.invoke({"celebrity": f"{my_input}"})
print(result)