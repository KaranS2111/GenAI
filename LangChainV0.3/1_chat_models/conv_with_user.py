from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat_history = [] #wrapper for knowing previous context

#First message should ne a system msg
system_msg = SystemMessage(content="You are an AI assistant.")
chat_history.append(system_msg) #adding system message to custom chat history

#Chat loop
while True:
    query = input("You: ") #asking the user to prompt a query
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) #Adding our msgs to history


    #Getting Ai responses based on chat history
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) #adding AI response in chat history

    print(f"AI: {response}")


print("-------CHAT HISTORY-------")
print(chat_history)