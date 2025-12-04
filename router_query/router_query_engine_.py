from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI

#loading documents
docs_exm_1 = SimpleDirectoryReader(r"../data/data1", required_exts=[".pdf"]).load_data()
docs_exm_2 = SimpleDirectoryReader(r"../data/data2", required_exts=[".pdf"]).load_data()
print(len(docs_exm_1))
print(len(docs_exm_2))

