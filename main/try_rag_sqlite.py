from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


directory="/proj/mounted/sqlite-email-files"
documents = SimpleDirectoryReader(input_dir=directory).load_data()

# Chunk the content
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
retr_model_name_or_path = "/proj/mounted/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
embed_model = HuggingFaceEmbedding(model_name=retr_model_name_or_path)
Settings.embed_model = embed_model
vector_index = VectorStoreIndex(nodes)

retr = vector_index.as_retriever(similarity_top_k=3)
prova = retr.retrieve("How can I connect to a database?")
for i, el in enumerate(prova):
    print(f"ELEMENT {i+1}")
    print(el.get_content())
    print("\n")

