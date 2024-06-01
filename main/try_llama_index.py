import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from transformers import BitsAndBytesConfig, AutoTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
from llama_index.tools.notion import NotionToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)


hf_token = os.getenv('HF_KEY')
notion_token = os.getenv('NOTION_KEY')
wolfram_token = os.getenv('WOLFRAM_APP_ID')
# retr_model_name_or_path = "/proj/mounted/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
retr_model_name_or_path = "BAAI/bge-base-en-v1.5"
infer_model_name_or_path = "/proj/mounted/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"

# WARNING: 4-BIT ENCODING DONE HERE, CONSIDER CHANGING (code from llamaindex llama 3 cookbook from)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    infer_model_name_or_path,
    token=hf_token,
)
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
llm = HuggingFaceLLM(
    model_name=infer_model_name_or_path,
    model_kwargs={
        "token": hf_token,
        # "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name=infer_model_name_or_path,
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

embed_model = HuggingFaceEmbedding(model_name=retr_model_name_or_path)
Settings.embed_model = embed_model
Settings.llm = llm

# def multiply(a: int, b: int) -> int:
#     """Multiply two integers and returns the result integer"""
#     return a * b

# def add(a: int, b: int) -> int:
#     """Add two integers and returns the result integer"""
#     return a + b

# def subtract(a: int, b: int) -> int:
#     """Subtract two integers and returns the result integer"""
#     return a - b

# multiply_tool = FunctionTool.from_defaults(fn=multiply)
# add_tool = FunctionTool.from_defaults(fn=add)
# subtract_tool = FunctionTool.from_defaults(fn=subtract)

all_tools = []
# all_tools.extend(YahooFinanceToolSpec().to_tool_list())
# all_tools.extend(WolframAlphaToolSpec(app_id=wolfram_token).to_tool_list())
wikipedia_tools = WikipediaToolSpec().to_tool_list()
all_tools.extend(LoadAndSearchToolSpec.from_defaults(wikipedia_tools[0], index_kwargs={'llm': None}).to_tool_list())
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=10)

query = "who is ben affleck married to?"
# get tools retrieved
# retrieved_tools_example = obj_retriever.retrieve(query)
# for i, tool in enumerate(retrieved_tools_example):
#     print(f"Metadata for tool {i}: {tool.metadata}")

agent = ReActAgent.from_tools(tool_retriever=obj_retriever, llm=llm, verbose=True)
# response = agent.chat(query)
# print(response)
# for llm querying only: response = llm.complete("How are you?" + tokenizer.eos_token)


