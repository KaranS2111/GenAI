import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

#Here we are using chromaDB vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"documents","A_01_Harry_Potter_and_the_Sorcerers_Stone.txt")
vectorstore = os.path.join(current_dir,"db","chroma_db")


print(vectorstore)
if not os.path.exists(vectorstore):
    print("Creating vector store for the first time")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file in {file_path} was missing.."
        )
    
    #Document Loading
    loader = TextLoader(file_path,encoding = 'UTF-8')
    documents = loader.load()

    # Chunking step
    text_splitter = CharacterTextSplitter(chunk_size = 900, chunk_overlap = 20)
    docs = text_splitter.split_documents(documents)

    print("\n------Chunk info------")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk example: \n{docs[0].page_content}")

    #Vector embedding process
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/text-embedding-004"
    )
    print("\n---------Finished Embedding ka kaam---------")


    #Storing embedding in vectorstore
    db = Chroma.from_documents(
        docs,embeddings,persist_directory=vectorstore)
    
    print("\n------Finished pehla padaav-------------")

else:

    print("\n-----------Mauja hi mauja----------------")

