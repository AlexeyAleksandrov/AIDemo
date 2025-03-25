import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def get_dataframe_from_file(filename: str):
    documents = []
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            documents.append({
                "id": i,
                "text": line.strip()
            })

    # создаем из наших документов датафрейм
    df = pd.DataFrame(documents)
    return df


def create_data_chroma_db(embeddings, filename: str, output_dir: str):
    # получаем датафрейм из строк файла
    df = get_dataframe_from_file(filename)

    # грузим фрейм в лоадер, выделив колонку для векторизации (здесь может быть место для дискуссий)
    loader = DataFrameLoader(df, page_content_column='text')
    data = loader.load()

    # разбивка данных
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    all_splits = text_splitter.split_documents(data)

    # векторизация данных
    Chroma.from_documents(documents=all_splits,
                          embedding=embeddings,
                          persist_directory=output_dir)

    print("Данные векторизованы!")
