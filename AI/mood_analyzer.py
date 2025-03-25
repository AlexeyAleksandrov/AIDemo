import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA


def check_mood(llm, embeddings, dialog: str) -> str:
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
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./mood_chroma_db")

    # формируем вопрос к модели
    question = "Какое настроение в данном диалоге у собеседников? " \
               "На сколько процентов, ты оцениваешь настроение по категориям: " \
               "восторженное, положительное, отрицательное, злобное? " \
               "Your answer must be in JSON: { mood: { восторженное: <value>, положительное: <value>, отрицательное: <value>, злобное: <value>} } where <value> " \
               "is value from 0 to 100% in percent"
    docs = vectorstore.similarity_search(question)

    # формирование результата
    qachain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    res = qachain.invoke({"query": question})
    text_result = res['result']

    return text_result
