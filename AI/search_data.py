from langchain_chroma import Chroma
from langchain.chains import RetrievalQA


def search_data_chroma_db(llm, embeddings, question: str, output_dir: str) -> str:
    # загрузка данных из БД
    vectorstore = Chroma(embedding_function=embeddings,
                         persist_directory=output_dir)

    # векторный поиск документов
    docs = vectorstore.similarity_search(question, k=1)

    # формирование результата
    res = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever()).invoke({"query": question})
    result_text = res['result']
    print('Ответ:' + result_text)

    return result_text

