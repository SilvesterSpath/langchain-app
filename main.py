import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
    )

# Use api_key when creating clients, e.g.:
# from openai import OpenAI
# client = OpenAI(api_key=api_key)
