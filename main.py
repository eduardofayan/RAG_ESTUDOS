from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()
CAMINHO_DB = "db"

prompt_template = """
Responda a seguinte pergunta do usuário:
{pergunta}

com base nessas informações:

{base_conhecimento}
"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    funcao_embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Carregar o Banco de Dados Vetorial
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embedding,
        )
    # Comparar a pergunta do usuario (embedding) com os embeddings da base de conhecimento (Banco de dados)
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)

    # Imprimir os resultados da busca de similaridade para depuração
    """ print(resultados) """

    # Imprimir a quantidade de resultados encontrados para depuração
    """ print(len(resultados)) """
    
    if len(resultados) == 0 or resultados[0][1] < 0.6:
        print("Não sei")
        return
    textos_resultados = []

    for resultado, score in resultados:
        texto = resultado.page_content
        textos_resultados.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultados)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta":pergunta, "base_conhecimento":base_conhecimento})
    
    # Exibir o prompt final que será enviado ao modelo
    """ print(prompt) """

    modelo = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )
    resposta = modelo.invoke(prompt)
    print("Resposta da IA:", resposta)
    print("Resposta da IA:", resposta.content)

perguntar()