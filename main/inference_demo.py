import os
import sys
import io
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from transformers import BitsAndBytesConfig, AutoTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.tools.notion import NotionToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
import gradio as gr


hf_token = os.getenv('HF_KEY')
notion_token = os.getenv('NOTION_KEY')
wolfram_token = os.getenv('WOLFRAM_APP_ID')
# retr_model_name_or_path = "/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
retr_model_name_or_path = "/proj/mounted/retr_model_toole_overlapping.pth"
infer_model_name_or_path = "/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"

# WARNING: 4-BIT ENCODING DONE HERE, CONSIDER CHANGING (code from llamaindex llama 3 cookbook from)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
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
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": False,
        "temperature": 0,
        "top_p": 0,
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
all_tools.extend(YahooFinanceToolSpec().to_tool_list())
all_tools.extend(WolframAlphaToolSpec(app_id=wolfram_token).to_tool_list())
wikipedia_tools = WikipediaToolSpec().to_tool_list()
wikipedia_wrapper = LoadAndSearchToolSpec.from_defaults(wikipedia_tools[0])
all_tools.extend(wikipedia_wrapper.to_tool_list())
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=1)

system_message = (
    "You are a model focused on function calling. You will be given a query and some tools to leverage. "
    "If you find it useful, you can call the tools to help answer the query. "
    "Just make sure to read the tools documentation well and understand the right parameters to pass."
)

prompt_template = (
    f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>'
    f'<|start_header_id|>user<|end_header_id|>\n\n{{}}<|eot_id|>'
    '<|start_header_id|>assistant<|end_header_id|>\n\n'
)
# get tools retrieved
# retrieved_tools_example = obj_retriever.retrieve("Who is Ben Affleck married to?")
# for i, tool in enumerate(retrieved_tools_example):
#     print(f"Metadata for tool {i}: {tool.metadata}")
# print(len(all_tools))
# agent = ReActAgent.from_tools(tool_retriever=obj_retriever, llm=llm, verbose=True)
# WITHOUT TOOL RETRIEVER
agent = ReActAgent.from_tools(all_tools, llm=llm, verbose=True)
# response = agent.chat(query)
# print(response)
# for llm querying only: response = llm.complete("How are you?" + tokenizer.eos_token)

def run_query(query):
    response = agent.chat(query)
    retrieved_tools = obj_retriever.retrieve(query)
    names = [tool.metadata.name for tool in retrieved_tools]
    return response, names

iface = gr.Interface(
    fn=run_query,
    inputs=gr.Textbox(label="Query Input"),
    outputs=[gr.Textbox(label="Chat Response"), gr.Textbox(label="Tool Names")],
).launch(share=True)




