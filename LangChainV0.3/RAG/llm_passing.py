import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore = os.path.join(current_dir,"db","chroma_db")

#Embedding model for user q
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

#Embedding model for user q
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
#same model that was used for embedding 
#private data

#Loading existing vectorstore with embeddings
db = Chroma(persist_directory=vectorstore,
            embedding_function=embeddings)


#Retriever
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    #Similarity search operation
    search_kwargs = {"k":5,"score_threshold":0.1},
    #k = top k chunks
)

#User query
query = "Name the groups in Hogwarts and which group was Harry in?"

relevant_chunks = retriever.invoke(query)

#kinda template
combined_input =  (
    "Here are some documents that might help answer the question: "
    + query
    +"\n\nRelevant Documents\n"
    +"\n\n".join([doc.page_content for doc in relevant_chunks])
    +"\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents. respond with 'I'm not sure'."
)

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n-----Response by AI-------")
print(result.content)