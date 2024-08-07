INSTRUCTION = """You are a function caller. Given a user query and the definition of a single API function, generate the appropriate function call. Return only the function call, using single quotes for strings and separating parameters with commas.

Example:
Function: add_reminder(text: str, date: str, time: str)
User: "Add a reminder to buy groceries tomorrow at 2 PM"
Response: add_reminder('Buy groceries', 'tomorrow', '2 PM')

Here's the API definition:
"""

TOOL_LLAMA_GROQ_PROMPT = """
You are a function calling AI model. You are provided with a function signature within <tools></tools> XML tags. Your task is to call the given function to assist with the user query. Don't make assumptions about what values to plug into functions. Only return the function call with a standard format FUNCTION_NAME(ARGS).

Here are the available tool:
"""

pseudo_name_mapping = {
    "llama3-8B" : "meta-llama/Meta-Llama-3-8B-Instruct",
    'codestral-22B' : "mistralai/Codestral-22B-v0.1",
    'gemma2-2B' : "google/gemma-2-2b-it",
    'groqLlama3Tool-8B' : "Groq/Llama-3-Groq-8B-Tool-Use"
}

pseudo_name_instr_mapping = {
    "llama3-8B" : INSTRUCTION,
    'codestral-22B' : INSTRUCTION,
    'gemma2-2B' : INSTRUCTION,
    'groqLlama3Tool-8B' : TOOL_LLAMA_GROQ_PROMPT
}


PROMPT_TEMPLATES = {
    'llama3-8B' : {
        'prompt_template' : (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n{api_def}'
            '<|eot_id|><|start_header_id|>user<|end_header_id|>{query}:<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    },
    'codestral-22B' : {
        'prompt_template' : (
            '<s>[INST] {instruction}\n{api_def}\n\nQuery: {query}'
            '[/INST] {answer}</s>'
        ),
        'answer_template' : '[/INST]'
    },
    'gemma2-2B' : {
      'prompt_template' : (
          '<bos><start_of_turn>user\n{instruction}\n{api_def}\n\n{query}<end_of_turn>\n'
          '<start_of_turn>model\n{answer}<end_of_turn>\n'
      ),
      'answer_template' : '<start_of_turn>model\n'
    },
    'groqLlama3Tool-8B' : {
         'prompt_template' : (
            '<|start_header_id|>system<|end_header_id|>{instruction}\n<tools>{api_def}</tools>'
            '<|eot_id|><|start_header_id|>user<|end_header_id|>{query}'
            '<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    }
}