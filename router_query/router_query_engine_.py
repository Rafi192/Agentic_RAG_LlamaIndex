from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
import logging
load_dotenv()
logger = logging.getLogger(__name__)

import sys
sys.stdout.reconfigure(encoding='utf-8')

# checking
llm_gemini = GoogleGenAI(model="gemini-2.5-flash")
# response = llm.complete("hello gemini")
# print(response)


#loading docs
doc_2024 = SimpleDirectoryReader(r"data").load_data()
doc_2008 = SimpleDirectoryReader(r"data_two").load_data()
if doc_2024:
    logger.info("directory is read")

else:
    logger.error("could not load documents")

#creating indexes
try:
    index_pdf = VectorStoreIndex.from_documents(doc_2024)
    index_2008 = VectorStoreIndex.from_documents(doc_2008)
    if index_pdf and index_2008:
        logger.info("Vector index created!")

except:
    logger.error("could not create a vector index")

#creating the query engine
engine_one = index_pdf.as_query_engine(similarity_top_k =3)
engine_two = index_2008.as_query_engine(similarity_top_k = 3)


#defining tools with description
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine_one,
        metadata={
            "name":"financial 2008",
            "Description": "contains financial crisis data of 2008"
        }
    ),
    QueryEngineTool(
        query_engine=engine_two,
        metadata={
            "name": "financial_2008",
            "description": "Financial crisis of 2008"
        }
    )
]

# creating router with LLM selector
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm_gemini),
    query_engine_tools=query_engine_tools,
    verbose=True
)

response = router_query_engine.query("what was the revenuew in Q3 2023?")
print(response)