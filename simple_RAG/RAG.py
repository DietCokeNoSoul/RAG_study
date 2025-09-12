from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI 
import os
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma

'''
    查询ChromaDB中的文本
'''
def query_collection(query):
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder='Models')
    chroma_client = chromadb.PersistentClient(path="../chroma_db")
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
    prompt = PromptTemplate(
        template="""你是一个严谨的RAG助手，请根据提供的上下文内容，准确且简洁地回答用户的问题。如果上下文信息不足以回答问题，请直接说:"根据上下文信息无法回答，请提供更多信息"，并且在回答后，附上使用了哪些上下文。
        上下文内容：
        {context}
        问题是：
        {query}
        """,
        input_variables=["context", "query"]
    )
    return prompt.format(context=context, query=query)
                
                    
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
