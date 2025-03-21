import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

def check_mood_gigachat(dialog: str) -> str:
    # Загрузка заранее обученной модели и токенизатора GigaChat
    model_name = "sberbank-ai/gigachat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Инициализация встраивания (embeddings) через HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Формируем DataFrame из текста
    df = pd.DataFrame([{
        'id': 0,
        'text': dialog
    }])

    # Загрузка документа из DataFrame
    documents = [Document(page_content=row['text'], metadata={"id": row['id']}) for _, row in df.iterrows()]

    # Разбиение текста на части для лучшего анализа
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    all_splits = text_splitter.split_documents(documents)

    # Создание хранилища векторов с использованием Chroma
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./mood_chroma_db")

    # Формируем запрос для анализа настроения
    question = "Какое настроение в данном диалоге у собеседников? На сколько процентов ты оцениваешь настроение по категориям: восторженное, положительное, отрицательное, злобное? Ответ должен быть в формате JSON: { mood: <значение> }, где <значение> одно из восторженное, положительное, отрицательное, злобное."

    # Поиск документов, связанных с вопросом
    docs = vectorstore.similarity_search(question)

    # Настраиваем пайплайн для обработки вопросов
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # Формируем результат на основе системы восстановления
    retriever = vectorstore.as_retriever()
    qachain = RetrievalQA.from_chain_type(llm=qa_pipeline, retriever=retriever)
    res = qachain.invoke({"query": question})
    text_result = res['result']

    return text_result
