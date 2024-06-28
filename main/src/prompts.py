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
    }
}

INSTRUCTIONS = {
    'tool_selection' : (
        "You are a helpful AI assistant. Your current task is to "
        "choose the appropriate tool to solve the user's query based "
        "on their question. I will provide you with the user's "
        "question and information about the tools. "
        "If there is a tool in the list that is applicable to this "
        "query, please return the name of the tool (you can only "
        "choose one tool). If there isn't, please return 'None.'. "
        "List of Tools with Names and Descriptions:"
    ),
    'function_calling' : (
        "Given a list of functions with their documentation, call the correct function "
        "with the correct parameters in the form function_name(parameter 1, parameter 2). "
        "Do not add any other text apart from the function call.\n"
        "Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk'). "
        "Here is the documentation of all the functions."
    )
    # 'function_calling' : (
    #     "You are a helpful AI assistant. Your current task is to call the correct function based on the user's query. "
    #     "Given a list of functions with their documentation, call the correct function "
    #     "with the correct parameters in the form function_name(parameter 1, parameter 2). "
    #     "Do not add any other text apart from the function call.\n"
    #     "Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk'). "
    #     "List of functions:"
    # )
}