from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer

'''
    对文本进行分句、嵌入并存储到ChromaDB中
'''
def embed_and_store():
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', cache_folder='Models')
    chroma_client = chromadb.PersistentClient(path="../chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    # 读取所有文本文件并进行分句
    path_list = list(Path("../AI").glob("**/*.txt"))
    content_list = []
    for path in path_list:
        content = path.read_text(encoding="utf-8")
        # 按行分句
        sentences = content.splitlines()
        print(AutoTokenizer.from_pretrained(sentences[0]))
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
    
if __name__ == "__main__":
    embed_and_store() # 只需运行一次存储