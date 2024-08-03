import os
import warnings
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error
import openai
import pandas as pd
import base64
import fitz
import google.generativeai as genai
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
from io import BytesIO
from docx import Document
from fpdf import FPDF

warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

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

def save_pdf_to_database(pdf_path):
    connection = connect_to_database()
    if connection is None:
        return None
    
    cursor = connection.cursor()
    with open(pdf_path, 'rb') as pdf_file:
        binary_pdf = pdf_file.read()
    
    try:
        cursor.execute("INSERT INTO pdfs (pdf_data) VALUES (%s)", (binary_pdf,))
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

def retrieve_pdf_from_database(pdf_id):
    connection = connect_to_database()
    if connection is None:
        return None
    
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT pdf_data FROM pdfs WHERE id = %s", (pdf_id,))
        result = cursor.fetchone()
        if result:
            pdf_data = result[0]
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"retrieved_{pdf_id}.pdf")
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(pdf_data)
            return pdf_path
        else:
            return None
    except Error as e:
        print(f"Error retrieving PDF from MySQL: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract VBA code from the uploaded Excel file
        vba_code = extract_vba_from_excel(file_path)
        
        # Analyze VBA code with GenAI and generate documentation
        documentation = generate_vba_documentation(vba_code)
        
        # Generate PDF report
        pdf_path = generate_pdf(documentation, filename="vba_documentation.pdf")
        
        # Save PDF to database
        pdf_id = save_pdf_to_database(pdf_path)
        
        return jsonify({'pdf_id': pdf_id}), 200
    else:
        return "Invalid file type", 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsm', 'xls'}

def extract_vba_from_excel(file_path):
    # Implement logic to extract VBA code from the uploaded Excel file
    pass

def generate_vba_documentation(vba_code):
    genai.configure(api_key='AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM')
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM", temperature=0.2, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM")
    
    template = """
    Analyze the following VBA code and generate a comprehensive documentation that includes logic, data flow, and process flow in plain English.
    
    VBA Code:
    {vba_code}
    
    Documentation:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=None,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    result = qa_chain({"query": vba_code})
    return result["result"]

def generate_pdf(content, file_name="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(file_name)
    return file_name

@app.route('/download/<int:pdf_id>', methods=['GET'])
def download_file(pdf_id):
    pdf_path = retrieve_pdf_from_database(pdf_id)
    if pdf_path:
        return send_file(pdf_path, as_attachment=True)
    else:
        return "PDF not found", 404

if __name__ == "__main__":
    app.run(debug=True)
