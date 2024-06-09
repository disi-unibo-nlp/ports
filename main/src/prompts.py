INSTRUCTION = '{}'
QUERY = '{}'
RESPONSE = '{}'
RETRIEVED_TEXT = '{}'

PROMPTS = {
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
    }
}
