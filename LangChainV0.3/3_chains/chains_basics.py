from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate #for step 2
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

#PROMPT TEMPLATE
messages = [
    ("system", "You are an expert who knows facts about planet {planet}."),
    ("human", "Tell me {fact_count} facts."),
] 

prompt_template = ChatPromptTemplate.from_messages(messages)

#Task 1 : Fill desired placeholders in template and create a final prompt
#Task 2 : Pass this template to LLM

#chaining eliminates the invoke invoke sequence
chain = prompt_template | llm | StrOutputParser()  # | is the pipe operator which is actually the chain for us
#Stroutputparser extracts .content part of the llm response so that 
# our llm response feels more real

#only 1 invoke method required now
#calling the final invoke() call
result = chain.invoke({"planet":"jupiter","fact_count":3})
print(result)