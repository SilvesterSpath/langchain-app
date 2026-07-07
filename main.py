import os

from dotenv import load_dotenv
from langchain_classic.chains import LLMChain, SequentialChain
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

test_prompt = PromptTemplate(
    template="""
    Test the following {language} code:\n {code}
    """,
    input_variables=["language", "code"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

full_chain = SequentialChain(chains=[code_chain, test_chain], input_variables=["language", "task"], output_variables=["code", "test"])
result = full_chain.invoke({"language": args.language, "task": args.task})

print(">>>>>>Code:<<<<<\n", result["code"])
print(">>>>>>Test:<<<<<\n", result["test"])
