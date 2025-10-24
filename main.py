
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_rag_pipeline() -> RetrievalQA:
    """Configura o pipeline de RAG com Llama-3 local."""
    # 1. Carrega PDFs
    loader = PyPDFDirectoryLoader("files/")
    docs = loader.load()
    if not docs:
        raise FileNotFoundError("Nenhum PDF encontrado em 'files/'.")
    logger.info(f"Carregados {len(docs)} PDFs.")

    # 2. Divide em chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Gerados {len(chunks)} chunks.")

    # 3. Cria embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Cria índice FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Índice FAISS criado.")

    # 5. Configura Llama-3
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Usa GPU se disponível, senão CPU
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 6. Cria chain RAG
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return qa

def main() -> None:
    """Executa o pipeline RAG e responde a uma pergunta."""
    try:
        # Configura o pipeline
        qa_chain = setup_rag_pipeline()
        
        # Pergunta de exemplo
        question = "Qual é o prazo de entrega no contrato X?"
        logger.info(f"Pergunta: {question}")
        
        # Resposta
        result = qa_chain.invoke({"query": question})  # Atualizado para .invoke
        answer = result["result"]
        sources = result["source_documents"]
        
        logger.info(f"Resposta: {answer}")
        logger.info("Fontes recuperadas:")
        for i, doc in enumerate(sources, 1):
            logger.info(f"Fonte {i}: {doc.page_content[:100]}... (Página {doc.metadata.get('page', 'N/A')})")
    
    except Exception as e:
        logger.error(f"Erro: {e}")
        exit(1)

if __name__ == "__main__":
    main()