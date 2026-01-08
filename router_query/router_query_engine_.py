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
print("হ্যালো, তুমি কেমন আছো?")
# llm = GoogleGenAI(model="gemini-2.5-flash")
# response = llm.complete("hello gemini")
# print(response)
print("প্রতিক্রিয়া সম্পূর্ণ")

#loading docs
doc_2024 = SimpleDirectoryReader(r"data").load_data()
if doc_2024:
    logger.info("directory is read")

else:
    logger.error("could not load documents")

print("বাংলা ঠিকমতো দেখা যাচ্ছে?")
print("আমি বাংলায় কথা বলি")
print("এইটা টেস্ট")
