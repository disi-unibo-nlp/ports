from transformers import AutoTokenizer
from huggingface_hub import login
import os
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
from src.prompts import PROMPT_TEMPLATES
login(token=os.getenv("HF_KEY"))
# models = [
#     ("/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45", "llama3"),
#     ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
#     ("mistralai/Codestral-22B-v0.1", "codestral"),
# ]

# for model, m_type in models:
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     INSTRUCTION = "instruction"
#     RETRIEVED_TEXT = "prova"
#     SYSTEM_MESSAGE = INSTRUCTION + " " + RETRIEVED_TEXT
#     QUERY = "query"
#     RESPONSE = "response"
#     msgs = [
#         {"role" : "system", "content" : SYSTEM_MESSAGE},
#         {"role" : "user", "content" : f"Query: {QUERY} Response:"},
#         {"role" : "assistant", "content" : RESPONSE}
#     ]
#     print(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) == PROMPT_TEMPLATES[m_type]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE))
#     print(f"Chat template: {tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}")
#     print(f'Prompt template: {PROMPT_TEMPLATES[m_type]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE)}')
#     print("-----------------")

INSTRUCTION = "instruction"
RETRIEVED_TEXT = "prova"
SYSTEM_MESSAGE = INSTRUCTION + " " + RETRIEVED_TEXT
QUERY = "query"
RESPONSE = "response"
msgs = [
    {"role" : "system", "content" : SYSTEM_MESSAGE},
    {"role" : "user", "content" : f"Query: {QUERY} Response:"},
    {"role" : "assistant", "content" : RESPONSE}
]

tokenizer = AutoTokenizer.from_pretrained("/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45")
print(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) == PROMPT_TEMPLATES["llama3"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE))
print(f"Chat template:\n{tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}")
print(f'Prompt template:\n{PROMPT_TEMPLATES["llama3"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE)}')
print("-----------------")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) == (PROMPT_TEMPLATES["phi3"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE) + '\n<|endoftext|>'))
print(f"Chat template:\n{tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}")
print(f'Prompt template:\n{PROMPT_TEMPLATES["phi3"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE)}')
print("-----------------")

msgs[1]["content"] = f"{SYSTEM_MESSAGE}\n\nQuery: {QUERY} Response:"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Codestral-22B-v0.1")
print(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) == PROMPT_TEMPLATES["codestral"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE))
print(f"Chat template:\n{tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}")
print(f'Prompt template:\n{PROMPT_TEMPLATES["codestral"]["prompt_template"].format(INSTRUCTION, RETRIEVED_TEXT, QUERY, RESPONSE)}')
print("-----------------")

# tokenizer = MistralTokenizer.v3()
# completion_request = ChatCompletionRequest(messages=[SystemMessage(content="Messaggio di sistema"), UserMessage(content="Hello")])
# text = tokenizer.encode_chat_completion(completion_request).text
# print(text)
