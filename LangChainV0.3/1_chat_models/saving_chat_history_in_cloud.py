from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

PROJECT_ID = "learning-ai-d63e7"
SESSION_ID = "new_user_session" #fictional user id (unique)
COLLECTION_NAME = "chat_history_1" #Name of the collection consisting of the documents
client = firestore.Client(project=PROJECT_ID) ##Iniitlizing client ??

##Initializing Firestore chat message history
print("Initializing chat message history......")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)

# Now instead of having history list from memory we are now gonna have it from cloud
# chat history now is being returned from langchain's firestore class

#same code

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

print("Start chatting with AI. Type 'exit' to quit interface.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")


#now we can store every single msg in cloud.