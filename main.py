import os

from dotenv import load_dotenv
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="Python")
parser.add_argument("--task", type=str, default="list of numbers")
args = parser.parse_args()

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
    )

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

code_prompt = PromptTemplate(
    template="""
    Write a very short {language} function that {task}.
    """,
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)

result = code_chain.run(language=args.language, task=args.task)

print(result)
