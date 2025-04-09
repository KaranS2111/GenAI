from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
#Runnables make each task a single reusable unit
#Each runnable lamda takes an input, does a task and gives the output
#PROMPT TEMPLATE
messages = [
    ("system", "You are an expert who knows facts about planet {planet}."),
    ("human", "Tell me {fact_count} facts."),
] 

prompt_template = ChatPromptTemplate.from_messages(messages)

#Runnables
#TASK1
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
#Gives our final desired template but does not invoke.
#I mean to say when we invoke the prompt template it turns it to a sendable format to LLM
#Here it does not and just fills the placeholder values

#TASK2
#Taking a prompt string from previous runnable and then pass it to LLM using invoke
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))

#Task3
#to convert to a human readable format
parse_output = RunnableLambda(lambda x: x.content)

#MANUAL CHAINING
chain = RunnableSequence(first=format_prompt, middle=[invoke_model]
                         ,last=parse_output)
#This is just chaining all runnables into a sequence
#RunnableSequence class only takes 3 parameters
#middle parameter can take n number of tasks in an array/list form

#So, either we use RunnableSequence or use 
#LCEL (LangChain Expression Language) which is the
#pipe operator to connect diff runnables together
#Both r same only (Runnables are not LCEL though)
response=chain.invoke({"planet":"Gargantua","fact_count":2})
print(response)



