from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

#RunnableParallel is the class that Langchain provides
#to run multiple chains parallely


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

#prompt template for movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)

#Plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

#Character analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

#Chains itself
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain =(
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

#combined chain
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot":plot_branch_chain,"characters": character_branch_chain})
    #two distinct chain branches
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"],x["branches"]["characters"]))
)

#Running the chain
result = chain.invoke({"movie_name":'Kuch Kuch Hota Hai'})
print(result)