# Requires transformers>=4.51.0
from sentence_transformers import CrossEncoder


def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

task = "Given a web search query, retrieve relevant passages that answer the query"

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


pairs = [
    [format_queries(query, task), format_document(doc)]
    for query, doc in zip(queries, documents)
]
scores = model.predict(pairs)
print(scores.tolist())
