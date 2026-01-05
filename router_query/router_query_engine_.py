from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

load_dotenv()

# checking
print("checking LLM for Gemini")
llm = GoogleGenAI(model="gemini-2.5-flash")
response = llm.complete("hello gemini")
print(response)
print("gemini response done !")