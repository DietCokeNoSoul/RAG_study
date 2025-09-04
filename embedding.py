from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import warnings
from openai import OpenAI 
import os

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

'''
    对文本进行分句、嵌入并存储到ChromaDB中
'''
def embed_and_store():
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder='Models')
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    # 读取所有文本文件并进行分句
    path_list = list(Path("AI").glob("**/*.md"))
    content_list = []
    for path in path_list:
        content = path.read_text(encoding="utf-8")
        # 按行分句
        sentences = content.splitlines()
        # 整体添加到列表中
        content_list.extend(sentences)

    # 生成嵌入并存储
    embeddings = model.encode(content_list, convert_to_tensor=True, truncate_dim=256)
    
    collection.add(
        ids=[str(i) for i in range(len(content_list))],
        embeddings=embeddings.tolist(),
        documents=content_list,
    )
    
    print(f"Stored {len(content_list)} sentences into the collection.")


'''
    查询ChromaDB中的文本
'''
def query_collection(query):
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder='Models')
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    
    query_embedding = model.encode(query, convert_to_tensor=True, truncate_dim=256)
    
    result = collection.query(
        include=["documents"],
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    return result.get("documents")[0]


'''
    检索
'''
def retrieve(query):
    context = ""
    for doc in query_collection(query):
        context += "\n" + doc + "\n"
        context += '-------------------------------'
    return context


'''
    增强Query
'''
def augmented(query, context=""):
    if not context:
        return f"请简要回答下面问题：{query}"
    
    else:
        return f"请根据上下文内容回答问题：{context}\n如果上下文信息不足以回答问题，请直接说:\"根据上下文信息无法回答，请提供更多信息\"\n问题是：{query}"
                
                    
'''
    生成回答
'''
def api(query):
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "什么是暴力搜索"
    context = retrieve(query)
    prompt = augmented(query, context)
    print("Prompt to LLM:")
    print(prompt)
    response = api(prompt)
    print(response)