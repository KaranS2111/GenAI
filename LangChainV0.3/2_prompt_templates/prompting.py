from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate #for step 2

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

#Step 1 : creating a human understandable template format
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

#Step 2 : converting prompt to langchain readable format
prompt_template = ChatPromptTemplate.from_template(template)

#print(prompt_template)
#Step 3 : replacing all placeholders manually
prompt = prompt_template.invoke({
    "tone": "exciting",
    "company": "rajshree printers",
    "position": "Son in law",
    "skill": "pyaar"
})

# print(prompt)

# #Invoke call
# result = llm.invoke(prompt)

# print(result.content)

#Moving forward
# What if we want a customized System message as well just like human msg above

#Human readable format
messages = [
    ("system", "You are a comedian who tells desi jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
] #List of tuples (both system and human msgs have tuples)

prompt_template = ChatPromptTemplate.from_messages(messages) #--> converting to langchain readable format )
#takes a simple string as well as a list of values

prompt = prompt_template.invoke({"topic": "boyfriend", "joke_count": 2})
result = llm.invoke(prompt)
print(result.content)
