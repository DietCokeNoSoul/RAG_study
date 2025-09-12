from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def process_pdfs_with_semantic_splitting(pdf_directory):
    """使用语义切分处理PDF文件"""
    
    # 加载PDF文件
    loader = DirectoryLoader(
        pdf_directory,  # 根目录
        glob="**/*.pdf", # 匹配所有子目录中的PDF文件
        loader_cls=PyPDFLoader # 指定对于每个文件的加载器,pyPDFLoader的基本单位是PDF页
    )
    documents = loader.load()
    
    # 设置嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name='Qwen/Qwen3-Embedding-0.6B',
        cache_folder='Models'
    )
    
    # 语义切分
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85  # 85%分位数作为阈值
    )
    
    chunks = semantic_chunker.split_documents(documents)
    
    print(f"处理了 {len(documents)} 个PDF文档")
    print(f"语义切分成 {len(chunks)} 个文本块")
    
    return chunks


def process_pdfs_from_directory(pdf_directory):
    """处理目录中的所有PDF文件（传统方式）"""
    
    # 加载PDF文件
    loader = DirectoryLoader(
        pdf_directory, 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"处理了 {len(documents)} 个PDF文档")
    
    # 文本切割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每块大小
        chunk_overlap=200,  # 重叠部分
        separators=["\n\n", "\n", " ", ""]  # 分割符优先级
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"切割成 {len(chunks)} 个文本块")
    
    return chunks


if __name__ == "__main__":
    # 示例使用
    pdf_chunks = process_pdfs_from_directory("Paper/2ATAKKQD")
    
    vector_store = Chroma.from_documents(
        pdf_chunks, 
        embedding=HuggingFaceEmbeddings(
            model_name='Qwen/Qwen3-Embedding-0.6B',
            cache_folder='Models'
        ), 
        persist_directory="./chroma_db",
        collection_name="pdf_collection"
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    relevant_docs = base_retriever.get_relevant_documents("what is Dynamic?")
