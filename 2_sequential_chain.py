from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

#os.environ['LANGSMITH_PROJECT']='demo-2'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name' : 'report_generation',
    'tags' : ['llm app', 'report_generation','summerization'],
    'metadata' : {'model_name':'google gemini'}
}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
