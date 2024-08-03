import warnings
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import gradio as gr
from docx import Document
import PyPDF2

warnings.filterwarnings("ignore")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n\n'.join(full_text)

# Function to extract text from TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding='utf-8') as f:
        return f.read()

file_paths = ["finalreport.pdf"]

# Extract text from files
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

# Initialize model and embeddings
model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM",
    temperature=0.1, 
    convert_system_message_to_human=True
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM")
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

# Create QA chain
template = """YOU are a vehicle assistant.Answer the questions based on the context provided
Give a detailed explanation for the questions in a conversational way. Use the context provided. If you don't know the answer, say so. Always say "Thanks for asking!" at the end.
Context:
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

# Load history from file if it exists
history_file = "history.json"

def load_history():
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  # Ensure that the data is a list
                    return data
            except json.JSONDecodeError:
                pass  # If there's an error decoding, return an empty list
    return []

def save_history(history):
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

history = load_history()

# Function to handle questions and maintain history
def ask_question(question):
    global history
    result = qa_chain({"query": question})
    answer = result["result"]
    history.append({"question": question, "answer": answer})
    save_history(history)
    # Format history for display
    history_md = ""
    for entry in history:
        history_md += f"*USER:* {entry['question']}\n\n*BOT:* {entry['answer']}\n\n---\n\n"
    return history_md

# Format history for initial display
initial_history_md = ""
if not history:
    initial_history_md = "Hey there! I'm your Inspector assistant. You can ask me questions related  and I'll help you. Let's get started!"
else:
    for entry in history:
        initial_history_md += f"*User:* {entry['question']}\n\n*Bot:* {entry['answer']}\n\n---\n\n"

# Create Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.HTML(
        """
        <style>
            .fixed-bottom {
                position: fixed;
                bottom: 0;
                width: 100%;
                padding: 10px;
                box-shadow: 0 -1px 10px rgba(0, 0, 0, 0.1);
               
            }
            .scrollable-history {
                max-height: 80vh;
                overflow-y: auto;
                margin-bottom: 100px; /* Space for the fixed bottom bar */
            }
        </style>
        """
    )
    
    # Markdown block to display history
    history_output = gr.Markdown(value=initial_history_md, elem_classes="scrollable-history")
    
    # Row for question input and submit button
    with gr.Row(elem_classes="fixed-bottom"):
        # Column for question input and submit button
        with gr.Column():
            # Text area for question input
            question_input = gr.Textbox(lines=2, placeholder="Type your question here...", show_label=False)
            # Submit button
            submit_button = gr.Button("Submit")
            # Function to handle submit action
            submit_button.click(ask_question, inputs=question_input, outputs=history_output)
            submit_button.click(lambda: "", None, question_input)  # Clears the input box
    
    # Display history above the question input and submit button pair
    history_output

demo.launch(share=True)
