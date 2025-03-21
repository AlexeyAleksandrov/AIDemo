import getpass
import os

import pandas as pd
import time

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_gigachat.chat_models import GigaChat


def get_giga_chat_llm():
    # инициализация GigaChat
    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

    return GigaChat(verify_ssl_certs=False)


def check_mood(dialog: str) -> str:
    # ollama = Ollama(
    #     base_url='http://localhost:11434',
    #     model="openchat",
    #     format="json"
    # )  # объект ollama
    ollama = get_giga_chat_llm()

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")  # запуск ollama embedded

    df = pd.DataFrame([{
        'id': 0,
        'text': dialog
    }])  # формирование датафрейма из текста

    # грузим фрейм в лоадер, выделив колонку для векторизации (здесь может быть место для дискуссий)
    loader = DataFrameLoader(df, page_content_column='text')
    data = loader.load()

    # разбивка данных
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    all_splits = text_splitter.split_documents(data)

    # векторизация данных
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory="./mood_chroma_db")

    # формируем вопрос к модели
    question = "Какое настроение в данном диалоге у собеседников? " \
               "На сколько процентов, ты оцениваешь настроение по категориям: " \
               "восторженное, положительное, отрицательное, злобное? " \
               "Your answer must be in JSON: { mood: <value> } where <value> " \
               "is восторженное, положительное, отрицательное, злобное"
    docs = vectorstore.similarity_search(question)

    # формирование результата
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    res = qachain.invoke({"query": question})
    text_result = res['result']

    return text_result
