from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "google-1.5-flash")


# main prompt templates
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# prompt template for translation to Hindi
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

#In this task we are getting facts related to animals and 
# showing them in French as output
# Let's start chaining.


# Defining additional processing steps using RunnableLambda
# count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "spanish"})
# prepare for translation is actually
#the input to translation template that will be passed to LLM
# stroutput parser gives .content string format but for translation task
# we need placeholder filling of the translation templae
#hence this runnable is required


# combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser() 

result = chain.invoke({"animal":"cat","count":2})

print(result)

#CODE NOT WORKING


