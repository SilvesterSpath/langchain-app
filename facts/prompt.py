from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(model="gpt-4o-mini"),
  chain_type="stuff",
  retriever=retriever
)

result = chain.run("What is an interesting fact about the English language?")
print(result)


