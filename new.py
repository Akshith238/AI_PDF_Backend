from flask import Flask, request, send_file, jsonify
import os
import fitz
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error
from docx import Document
import PyPDF2
import warnings
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")

def func():
    # MySQL Configuration
    mysql_config = {
        'host': 'localhost',
        'database': 'caterpillar',
        'user': 'root',
        'password': 'root'
    }

    def connect_to_database():
        try:
            connection = mysql.connector.connect(**mysql_config)
            if connection.is_connected():
                return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
        return None

    def save_pdf_to_database(pdf_path, repstatus):
        connection = connect_to_database()
        if connection is None:
            return None
        
        cursor = connection.cursor()
        with open(pdf_path, 'rb') as pdf_file:
            binary_pdf = pdf_file.read()
        
        try:
            cursor.execute("INSERT INTO pdfs (pdf_data, repstatus) VALUES (%s, %s)", (binary_pdf, repstatus))
            connection.commit()
            pdf_id = cursor.lastrowid
        except Error as e:
            print(f"Error inserting PDF into MySQL: {e}")
            connection.rollback()
            pdf_id = None
        finally:
            cursor.close()
            connection.close()
        
        return pdf_id

    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text

    def extract_text_from_docx(docx_path):
        doc = Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n\n'.join(full_text)


    file_paths = ["finalreport.pdf","data/Lists.docx"]
    texts = []
    for path in file_paths:
        if path.endswith(".pdf"):
            texts.append(extract_text_from_pdf(path))
        elif path.endswith(".docx"):
            texts.append(extract_text_from_docx(path))
        elif path.endswith(".txt"):
            texts.append(extract_text_from_txt(path))

    context = "\n\n".join(texts)

    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_text(context)
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM",
                                temperature=0.2,convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )


    template = """
    You are provided with a truck inspection report and a checklist of items that need to be inspected. Your task is to compare the report with the checklist and identify any missing sections or subsections. If all sections are inspected, return 'None'.

    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    question = """
    Identify any missing sections or subsections from the checklist.
    """
    result = qa_chain({"query": question})
    ans=str(result["result"])
    print(ans)


    if 'None' in ans:
        bol = False
    else:
        bol = True

    # Determine the repstatus based on the result
    if bol:
        repstatus = "pending"
    else:
        repstatus = "needs_to_be_verified"

    # Save the PDF to the database and get the pdf_id
    pdf_id = save_pdf_to_database("finalreport.pdf", repstatus)

    # Insert into pendingrep table if there are missing sections
    if bol:
        connection = connect_to_database()
        if connection is not None:
            cursor = connection.cursor()
            try:
                cursor.execute("INSERT INTO pendingrep (id, data) VALUES (%s, %s)", (pdf_id, ans))
                connection.commit()
            except Error as e:
                print(f"Error inserting into pendingrep: {e}")
                connection.rollback()
            finally:
                cursor.close()
                connection.close()