from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_document_loader(file_path):
    logger.info(f"Getting document loader for file: {file_path}")
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension in ['.doc', '.docx']:
        return UnstructuredWordDocumentLoader(file_path)
    elif file_extension in ['.ppt', '.pptx']:
        return UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def process_chunk(chunk: str, llm: ChatGroq, prompt: PromptTemplate) -> str:
    """Process a single chunk of text to generate questions."""
    try:
        # Clean the chunk before processing
        cleaned_chunk = clean_text(chunk)
        if not cleaned_chunk:
            return ""

        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=False
        )
        return chain.run([Document(page_content=cleaned_chunk)])
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""

def file_processing(file_path):
    try:
        logger.info(f"Starting file processing for: {file_path}")
        loader = get_document_loader(file_path)
        logger.info("Loading document...")
        data = loader.load()
        logger.info(f"Document loaded successfully. Number of pages/sections: {len(data)}")

        # Combine all text and clean it
        question_gen = ' '.join(clean_text(page.page_content) for page in data)
        
        # Use smaller chunks for faster processing
        splitter = CharacterTextSplitter(
            chunk_size=2000,  # Reduced from 4000
            chunk_overlap=100,  # Reduced from 200
            separator="\n"
        )

        chunks = splitter.split_text(question_gen)
        logger.info(f"Text split into {len(chunks)} chunks")

        return chunks
    
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        raise

def llm_pipeline(chunks: List[str], file_path: str) -> Tuple[str, List[dict]]:
    try:
        logger.info("Starting LLM pipeline...")

        # Initialize LLM with optimized settings
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            max_tokens=300  # Reduced from 500
        )

        # Create prompt template
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )

        # Process chunks in parallel with fewer workers
        logger.info("Generating questions in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced from 3
            futures = [executor.submit(process_chunk, chunk, llm, prompt) for chunk in chunks]
            results = [f.result() for f in futures]

        # Combine and filter questions
        all_questions = []
        for result in results:
            questions = result.split('\n')
            all_questions.extend([q.strip() for q in questions if q.strip() and (q.strip().endswith('?') or q.strip().endswith('.'))])

        # Get top 10 questions
        display_questions = all_questions[:10]
        logger.info(f"Generated {len(all_questions)} questions, displaying top {len(display_questions)}")

        # Initialize answer generation with optimized settings
        logger.info("Initializing answer generation...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Force CPU usage for stability
        )
        
        # Create smaller chunks for answer generation
        splitter = CharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=50,  # Reduced from 100
            separator="\n"
        )
        
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(documents, embeddings)

        answer_llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            max_tokens=200  # Reduced from 300
        )

        answer_chain = RetrievalQA.from_chain_type(
            llm=answer_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 1})  # Reduced from 2
        )

        # Generate answers for display questions
        qa_list = []
        for question in display_questions:
            try:
                answer = answer_chain.run(question)
                qa_list.append({
                    "question": question,
                    "answer": answer
                })
            except Exception as e:
                logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
                qa_list.append({
                    "question": question,
                    "answer": "Error generating answer."
                })

        # Generate CSV file
        output_path = Path("static/output") / f"{Path(file_path).stem}_QA.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            import csv
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Question", "Answer"])
            
            # Write all questions and answers to CSV
            for question in all_questions:
                try:
                    answer = answer_chain.run(question)
                    csv_writer.writerow([question, answer])
                except Exception as e:
                    logger.error(f"Error generating answer for CSV: {question}. Error: {str(e)}")
                    csv_writer.writerow([question, "Error generating answer."])

        return str(output_path), qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise