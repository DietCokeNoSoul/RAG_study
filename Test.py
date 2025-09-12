from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import time
import torch

# 验证numpy版本兼容性
try:
    import numpy as np
    import scipy
    print(f"✅ NumPy version: {np.__version__}")
    print(f"✅ SciPy version: {scipy.__version__}")
    
    # 检查版本兼容性
    numpy_version = tuple(map(int, np.__version__.split('.')))
    if numpy_version >= (1, 22, 4) and numpy_version < (2, 3, 0):
        print("✅ NumPy版本符合要求 (>=1.22.4, <2.3.0)")
    else:
        print(f"⚠️ NumPy版本可能不兼容: {np.__version__}")
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 版本检查错误: {e}")

print("\n开始加载向量库...")
start_time = time.time()

# 检查设备可用性
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda:0'
else:
    print("未检测到 CUDA，将使用 CPU")
    device = 'cpu'

persist_dir = "./chroma_db"
collection_name = "pdf_collection"

# 全局模型缓存：只在首次运行时加载模型
if 'embedding' not in globals():
    print("首次加载嵌入模型...")
    model_start = time.time()
    
    try:
        embedding = HuggingFaceEmbeddings(
            model_name='Qwen/Qwen3-Embedding-0.6B',
            cache_folder='Models',
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ 嵌入模型加载完成 (设备: {device})，耗时: {time.time() - model_start:.2f}秒")
    except Exception as e:
        print(f"⚠️ 嵌入模型加载遇到问题: {e}")
        print("尝试使用CPU模式重新加载...")
        
        # 回退到CPU模式
        embedding = HuggingFaceEmbeddings(
            model_name='Qwen/Qwen3-Embedding-0.6B',
            cache_folder='Models',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ 嵌入模型加载完成 (CPU模式)，耗时: {time.time() - model_start:.2f}秒")
else:
    print("使用已缓存的嵌入模型 ✓")

# 检查向量库是否存在
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("发现已存在的向量库，直接加载...")
    load_start = time.time()
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=collection_name
    )
    
    print(f"✅ 向量库加载完成，耗时: {time.time() - load_start:.2f}秒")
    print(f"总耗时: {time.time() - start_time:.2f}秒")