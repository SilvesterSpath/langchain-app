from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug
from dotenv import load_dotenv

set_debug(True)

load_dotenv()

embeddings = OpenAIEmbeddings()
db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(),
  chain_type="refine",
  # chain_type="map_reduce",
  # chain_type="stuff",
  retriever=retriever,
)

result = chain.invoke({"query": "What is an interesting fact about the English language?"})
print(result["result"])


