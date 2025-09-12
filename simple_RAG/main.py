from simple_RAG.RAG import retrieve, augmented, api
from simple_RAG.store_embedding import embed_and_store
import warnings

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

if __name__ == "__main__":
    # embed_and_store() # 只需运行一次存储
    query = "什么是余弦相似度？"
    context = retrieve(query)
    prompt = augmented(query, context)
    print("Prompt to LLM:")
    print(prompt)
    response = api(prompt)
    print(response)