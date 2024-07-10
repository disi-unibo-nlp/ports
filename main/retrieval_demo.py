from transformers import (
    AutoTokenizer, 
    AutoModel, 
    DataCollatorWithPadding,
    set_seed
)
from datasets import load_from_disk
import torch
import pandas as pd
import torch.nn.functional as F
import gradio as gr

def main():
  
    # load models and tokenizers
    set_seed(42)
    retr_tokenizer = AutoTokenizer.from_pretrained("/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/")
    base_model = AutoModel.from_pretrained("/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/").to("cuda")
    finetuned_model = AutoModel.from_pretrained("/proj/mounted/toole_diff_docs_non_overlap").to("cuda")

    base_model.eval()
    finetuned_model.eval()

    def compute_embeddings(model, documents):
        # Compute token embeddings
        documents = {k: v.to("cuda") for k, v in documents.items()}
        model_output = model(**documents)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings_normal = F.normalize(sentence_embeddings, p=2, dim=1)
        del documents, model_output, sentence_embeddings
        return sentence_embeddings_normal

    with open("/proj/mounted/func-docs/documentation-toole-eval.txt", "r") as f:
        func_text = f.read()
    eval_documents = func_text.split("\n")
    tokenized_eval_documents = retr_tokenizer(eval_documents, padding=True, truncation=True, return_tensors='pt')

    def tokenize_function(samples):
        return retr_tokenizer(samples["query"], padding=True, truncation=True, return_tensors='pt')

    dataset = load_from_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")

    base_doc_embeddings = compute_embeddings(base_model, tokenized_eval_documents)
    finetuned_doc_embeddings = compute_embeddings(finetuned_model, tokenized_eval_documents)

    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    k = 3

    def get_top_k_docs_per_query(embedded_documents, batch, k, model):
        """
        Compute the top-k documents for each query in the batch, based on their cosine similarity
        """
        embedded_queries = compute_embeddings(model, batch)
        embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
        embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
        cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
        top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
        return top_k_docs.indices, top_k_docs.values

    verify_relevancy = lambda response, doc: response == doc.split(":")[0]

    def get_relevant_docs(response, documents):
        """
        For each example in the batch, retrieve its relevant document
        """        
        for i, doc in enumerate(documents):
                if verify_relevancy(response, doc):
                    return i, doc
        return None

    def my_function(query):
        query_match = lambda example: example['query'] == query
        filtered_dataset = dataset["test"].filter(query_match)
        assert(len(filtered_dataset) == 1)
        response = filtered_dataset["response"][0]
        rel_doc, doc_text= get_relevant_docs(response, eval_documents)
        tokenized_query = retr_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
        collated_query = data_collator(tokenized_query)
        docs_per_query_base, sim_per_query_base = get_top_k_docs_per_query(base_doc_embeddings, collated_query, k, base_model)
        docs_per_query_finetuned, sim_per_query_finetuned = get_top_k_docs_per_query(finetuned_doc_embeddings, collated_query, k, finetuned_model)
        docs_per_query_base = [eval_documents[i].split(":")[0] for i in docs_per_query_base[0].tolist()]
        docs_per_query_finetuned = [eval_documents[i].split(":")[0] for i in docs_per_query_finetuned[0].tolist()]
        sim_per_query_base = [round(num, 3) for num in sim_per_query_base[0].tolist()]
        sim_per_query_finetuned = [round(num, 3) for num in sim_per_query_finetuned[0].tolist()]
        dataframe_bge = pd.DataFrame(
            {
                "functions": docs_per_query_base,
                "similarities": sim_per_query_base,
            }
        )
        dataframe_finetuned = pd.DataFrame(
            {
                "functions": docs_per_query_finetuned,
                "similarities": sim_per_query_finetuned,
            }
        )
        print(dataframe_bge)
        return (
            bar_plot_fn(dataframe_finetuned, is_finetuned=True),
            bar_plot_fn(dataframe_bge),
            doc_text
        )

    def bar_plot_fn(simple, is_finetuned=False):
        return gr.BarPlot(
            simple,
            x="functions",
            y="similarities",
            title="Retrieved Tools by Fine-tuned Encoder (Ours)✅" if is_finetuned else "Retrieved Tools by BGE-base",
            tooltip=["functions", "similarities"],
            y_lim=[0,1],
            show_label=False,
            vertical=False,
        )

    def clear_all():
        return "", "", gr.BarPlot(label="Retrieved Tools by Fine-tuned Encoder (Ours)"), gr.BarPlot(label="Retrieved Tools by BGE-base")


    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {font-size: 2rem}") as demo:
        gr.HTML('<h1 align="center">Retrievers comparison</h1>')
        query = gr.Textbox(label="Query", placeholder="Enter your query here")
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", scale=2)
            clear_btn = gr.Button("Clear")
        with gr.Row():
            with gr.Column():
                ours = gr.BarPlot(label="Retrieved Tools by Fine-tuned Encoder (Ours)")
            with gr.Column():
                bge = gr.BarPlot(label="Retrieved Tools by BGE-base")
        correct_tool = gr.Textbox(label="Ground Truth Tool", placeholder="Correct tool's documentation will be displayed here")
        clear_btn.click(clear_all, outputs=[query, correct_tool, ours, bge])
        submit_btn.click(my_function, inputs=[query], outputs=[ours, bge, correct_tool])
        query.submit(my_function, inputs=[query], outputs=[ours, bge, correct_tool])
    demo.launch(share=True)

if __name__ == "__main__":
    main()

    # call("Can you show me the latest tweets about the #Olympics2021?")
    # iface = gr.Interface(
    #     fn=my_function,
    #     inputs=gr.Textbox(label="Query"),
    #     outputs=[
    #         gr.Textbox(label="Retrieved Tools by BGE-base"),
    #         gr.Textbox(label="Retrieval Similarities by BGE-base"),
    #         gr.Textbox(label="Retrieved Tools by Fine-tuned Encoder **(Ours)**"),
    #         gr.Textbox(label="Retrieval Similarities by Fine-tuned Encoder **(Ours)**"),
    #         gr.Textbox(label="Ground Truth Tool"),
    #     ],
    #     theme=gr.themes.Soft(),
    #     allow_flagging="never",
    # ).launch(share=True)