import getpass
import os

from langchain_gigachat.chat_models import GigaChat
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def get_giga_chat_llm():
    # инициализация GigaChat
    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

    return GigaChat(verify_ssl_certs=False)


def search_data_chroma_db(question: str, output_dir: str) -> str:
    print("Загрузка Ollama")
    # ollama = Ollama(
    #     base_url='http://localhost:11434',
    #     model="openchat"
    # )

    ollama = get_giga_chat_llm()

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

    # загрузка данных из БД
    vectorstore = Chroma(embedding_function=oembed, persist_directory=output_dir)

    # print("Введите вопрос:")
    # question = input()
    docs = vectorstore.similarity_search(question)

    # формирование результата
    res = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever()).invoke({"query": question})
    result_text = res['result']
    print('Ответ:' + result_text)

    return result_text

