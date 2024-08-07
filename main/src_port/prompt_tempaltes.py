INSTRUCTION = '{}'
QUERY = '{}'
RESPONSE = '{}'
RETRIEVED_TEXT = '{}'

PROMPT_TEMPLATES = {
    'llama3' : {
        'prompt_template' : (
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION} {RETRIEVED_TEXT}'
            f'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: {QUERY} Response:<|eot_id|>'
            f'<|start_header_id|>assistant<|end_header_id|>\n\n{RESPONSE}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>\n\n'
    },
    'phi3' : {
        'prompt_template' : (
            f'<|system|>\n{INSTRUCTION} {RETRIEVED_TEXT}<|end|>\n'
            f'<|user|>\nQuery: {QUERY} Response:<|end|>\n'
            f'<|assistant|>\n{RESPONSE}<|end|>'
        ),
        'answer_template' : '<|assistant|>\n'
    },
    'mixtral' : {
        'prompt_template' : (
            f'<|im_start|>system\n{INSTRUCTION} {RETRIEVED_TEXT}<|im_end|>\n'
            f'<|im_start|>user\nQuery: {QUERY} Response:<|im_end|>\n'
            f'<|im_start|>assistant\n{RESPONSE}<|im_end|>'
        ),
        'answer_template' : '<|im_start|>assistant\n'
    },
    'gemma' : {
      'prompt_template' : (
          f'<bos><start_of_turn>user\n{INSTRUCTION}\n{RETRIEVED_TEXT}\n\n{QUERY}<end_of_turn>\n'
          f'<start_of_turn>model\n{RESPONSE}<end_of_turn>\n'
      ),
      'answer_template' : '<start_of_turn>model\n'
    },
    'gpt2' : {
        'prompt_template' : (
            f'### Instruction:\n{INSTRUCTION} {RETRIEVED_TEXT}\n'
            f'### Query:\n{QUERY}\n'
            f'### Answer:\n{RESPONSE}'
        ),
        'answer_template' : '### Answer:\n'
    },
    'codestral' : {
        "TODO"
    }
}


INSTRUCTION = """You are a function caller. Given a user query and the definition of a single API function, generate the appropriate function call. Return only the function call, using single quotes for strings and separating parameters with commas.

Example:
Function: add_reminder(text: str, date: str, time: str)
User: "Add a reminder to buy groceries tomorrow at 2 PM"
Response: add_reminder('Buy groceries', 'tomorrow', '2 PM')


"""