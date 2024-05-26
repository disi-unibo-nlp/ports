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

hf_token = os.getenv('HF_KEY')

# WARNING: 4-BIT ENCODING DONE HERE, CONSIDER CHANGING (code from llamaindex llama 3 cookbook from)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
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
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)

all_tools = [multiply_tool, add_tool, subtract_tool]

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

query = "What is 90+(17*2)? Calculate step by step."
retrieved_tools_example = obj_retriever.retrieve(query)
# for i, tool in enumerate(retrieved_tools_example):
#     print(f"Metadata for tool {i}: {tool.metadata}")

agent = ReActAgent.from_tools(tool_retriever=obj_retriever, llm=llm, verbose=True)

response = agent.chat(query)

print(response)
# response = llm.complete("How are you?" + tokenizer.eos_token)


