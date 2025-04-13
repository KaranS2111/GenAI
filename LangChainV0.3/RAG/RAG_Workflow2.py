import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore = os.path.join(current_dir,"db","chroma_db")

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
    search_kwargs = {"k":5,"score_threshold":0.5},
    #k = top k chunks
)

#User query
query = "Who were playing in the quidditch match?"

relevant_chunks = retriever.invoke(query)

#Displaying relevant results
print("\n------ Relevant documents --------")
for i, doc in enumerate(relevant_chunks,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','Unknown')}\n")
