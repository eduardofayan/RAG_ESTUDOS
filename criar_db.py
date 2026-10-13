from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

# Função para carregar documentos da pasta base
def carregar_documentos():
    # ler todos os documentos pdf da pasta base
    carregador_documentos = PyPDFDirectoryLoader(PASTA_BASE)
    
    # carregar os documentos com .load da biblioteca PyPDFLoader
    documentos = carregador_documentos.load()

    # verificar se os documentos foram carregados corretamente
    print(f"{len(documentos)} documentos carregados.")

    # retornar os documentos carregados
    print(f"Documentos carregados: {documentos}")

    # retornar os documentos carregados
    return documentos

# Função para dividir documentos em chunks (partes menores da base)
def dividir_chunks(documentos):
    # criar o separador de documentos com tamanho de chunk e overlap (sobreposição) definidos
    separador_documento = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function = len,
        add_start_index=True
    )
    # dividir os documentos em chunks usando o separador criado
    chunks = separador_documento.split_documents(documentos)

    # exibir a quantidade de chunks gerados
    print(f"{len(chunks)} chunks gerados.")

    # retornar os chunks gerados a partir dos documentos
    return chunks

# Função para vetorizar os chunks gerados a partir dos documentos
def vetorizar_chunks(chunks):

    # criar os embeddings usando OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # criar a base de dados vetorial a partir dos chunks usando Chroma e OllamaEmbeddings
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")

criar_db()