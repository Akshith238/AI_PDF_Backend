import os
import warnings
import json
from fpdf import FPDF
import google.generativeai as genai
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from oletools.olevba import VBA_Parser

warnings.filterwarnings("ignore")

def extract_vba_from_excel(file_path):
    if not file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
        raise ValueError("The provided file is not an Excel file.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    vba_code = ""

    # Parse the Excel file to extract VBA macros
    vba_parser = VBA_Parser(file_path)
    if vba_parser.detect_vba_macros():
        for (filename, stream_path, vba_filename, vba_code_chunk) in vba_parser.extract_macros():
            vba_code += vba_code_chunk

    vba_parser.close()

    return vba_code

def analyze_macro(macro_code):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_text(macro_code)

    genai.configure(api_key='AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM')
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key='AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM', temperature=0.2, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyCqEKwd23ztVuk-dkCXypjeHWlcs41aCSM')
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    
    template = """
    You are the automated VBA Macro analyzer that evaluates the quality and efficiency of VBA macros, identifying potential inefficiencies, redundant code, and optimization opportunities. Generate report for a macro only once in the output.
    VBA Code:
    {context}
    Name: *Macro Name*
    Time Complexity:
    Efficiency:
    Redundant Code:
    Optimization Opportunities:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    question = "Analyze the VBA code for time complexity, efficiency, redundant code, and optimization opportunities."
    result = qa_chain({"query": question})
    return result["result"]

def parse_analysis_result(analysis_result):
    sections = analysis_result.split("\n")
    
    macro_analysis = {
        "time_complexity": "",
        "efficiency": "",
        "redundant_code": "",
        "optimization_opportunities": ""
    }
    
    for section in sections:
        if "Time Complexity:" in section:
            macro_analysis["time_complexity"] = section.split("Time Complexity:")[1].strip()
        elif "Efficiency:" in section:
            macro_analysis["efficiency"] = section.split("Efficiency:")[1].strip()
        elif "Redundant Code:" in section:
            macro_analysis["redundant_code"] = section.split("Redundant Code:")[1].strip()
        elif "Optimization Opportunities:" in section:
            macro_analysis["optimization_opportunities"] = section.split("Optimization Opportunities:")[1].strip()
    
    return macro_analysis

def generate_json(analysis_results):
    analysis_json = json.dumps(analysis_results, indent=4)
    return analysis_json

def generate_pdf(analysis_results, file_name="vba_analysis_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add title page
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Automated VBA Macro Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Generated Report", ln=True, align='C')
    pdf.ln(10)

    # Add macro analyses
    for i, analysis in enumerate(analysis_results):
        if i > 0:
            pdf.add_page()
        pdf.set_font("Arial", 'B', size=14)
        pdf.multi_cell(0, 10, f"Macro {i + 1} Analysis", align='L')
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        for key, value in analysis.items():
            pdf.multi_cell(0, 10, f"{key.replace('_', ' ').title()}: {value}")
            pdf.ln(5)
    
    pdf.output(file_name)
    return file_name

def main():
    file_path = "data/Book1.xlsm"  
    
    vba_code = extract_vba_from_excel(file_path)
    
    function_blocks = vba_code.split('End Sub')
    macro_analyses = []

    for function_block in function_blocks:
        if not function_block.strip():
            continue

        function_block += 'End Sub'
        
        # Generate documentation from VBA code
        documentation = analyze_macro(function_block)
        
        # Analyze the macro for time complexity, efficiency, redundant code, and optimization opportunities
        parsed_analysis = parse_analysis_result(documentation)
        macro_analyses.append(parsed_analysis)

    # Generate JSON report with analysis results
    analysis_json = generate_json(macro_analyses)
    print(f"\nJSON generated: {analysis_json}")

    # Generate PDF report with documentation and time complexity analysis
    pdf_path = generate_pdf(macro_analyses, file_name="vba_analysis_report.pdf")
    print(f"\nPDF generated at: {pdf_path}")

if __name__ == "__main__":
    main()
