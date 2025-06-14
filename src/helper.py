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
import time
from itertools import chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Rate limiting settings
RATE_LIMIT_DELAY = 0.5  # Reduced delay between API calls
MAX_RETRIES = 3
BATCH_SIZE = 5  # Process questions in batches

def get_document_loader(file_path):
    logger.info(f"Getting document loader for file: {file_path}")
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Error in get_document_loader: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def process_batch(questions: List[str], answer_chain: RetrievalQA) -> List[dict]:
    """Process a batch of questions together."""
    results = []
    for question in questions:
        try:
            answer = answer_chain.run(question)
            results.append({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
            results.append({
                "question": question,
                "answer": "Error generating answer. Please try again later."
            })
    time.sleep(RATE_LIMIT_DELAY)  # Single delay for the whole batch
    return results

def process_chunk_with_retry(chunk: str, llm: ChatGroq, prompt: PromptTemplate, max_retries: int = MAX_RETRIES) -> str:
    """Process a single chunk of text with retry logic."""
    for attempt in range(max_retries):
        try:
            cleaned_chunk = clean_text(chunk)
            if not cleaned_chunk:
                logger.warning("Empty chunk after cleaning")
                return ""

            logger.info(f"Processing chunk (attempt {attempt + 1}/{max_retries})")
            chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )
            
            # Add more context to the prompt
            formatted_prompt = prompt.format(text=cleaned_chunk)
            logger.info(f"Formatted prompt length: {len(formatted_prompt)}")
            
            result = chain.invoke([Document(page_content=cleaned_chunk)])
            
            # Handle dictionary response
            if isinstance(result, dict):
                if 'output_text' in result:
                    result = result['output_text']
                elif 'text' in result:
                    result = result['text']
                else:
                    logger.warning(f"Unexpected dictionary format: {result}")
                    result = str(result)
            
            if not result or not result.strip():
                logger.warning(f"Empty result received from LLM on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    return ""
                continue
                
            logger.info(f"Successfully processed chunk (attempt {attempt + 1})")
            return result
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to process chunk after {max_retries} attempts: {str(e)}")
                return ""
            logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))

def file_processing(file_path):
    try:
        logger.info(f"Starting file processing for: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")
            
        logger.info(f"File size: {file_size} bytes")
        
        # Get document loader
        loader = get_document_loader(file_path)
        logger.info("Document loader created successfully")
        
        # Load document
        logger.info("Loading document...")
        data = loader.load()
        
        if not data:
            raise ValueError("No content extracted from document")
            
        logger.info(f"Document loaded successfully. Number of pages/sections: {len(data)}")
        
        # Log first few characters of content for debugging
        sample_content = ' '.join(page.page_content[:100] for page in data[:2])
        logger.info(f"Sample content from first two pages: {sample_content}")

        # Combine all text and clean it
        question_gen = ' '.join(clean_text(page.page_content) for page in data)
        
        if not question_gen.strip():
            raise ValueError("No text content after cleaning")
            
        # Use larger chunks for faster processing
        splitter = CharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separator="\n"
        )

        chunks = splitter.split_text(question_gen)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("No chunks generated after text splitting")

        return chunks
    
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        raise Exception(f"Document analysis failed: {str(e)}")

def llm_pipeline(chunks: List[str], file_path: str) -> Tuple[str, List[dict]]:
    try:
        logger.info("Starting LLM pipeline...")

        # Initialize LLM with optimized settings
        try:
            logger.info("Initializing LLM...")
            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="gemma2-9b-it",
                temperature=0.7
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

        # Create prompt template
        try:
            logger.info("Creating prompt template...")
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["text"]
            )
            logger.info("Prompt template created successfully")
        except Exception as e:
            logger.error(f"Failed to create prompt template: {str(e)}")
            raise Exception(f"Failed to create prompt template: {str(e)}")

        # Process chunks in parallel
        logger.info("Generating questions in parallel...")
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_chunk_with_retry, chunk, llm, prompt) for chunk in chunks]
                results = [f.result() for f in futures]
            logger.info(f"Processed {len(results)} chunks")
        except Exception as e:
            logger.error(f"Failed to process chunks in parallel: {str(e)}")
            raise Exception(f"Failed to process chunks: {str(e)}")

        # Combine and filter questions
        all_questions = []
        for i, result in enumerate(results):
            if not result:
                logger.warning(f"Empty result for chunk {i}")
                continue
            questions = result.split('\n')
            valid_questions = [q.strip() for q in questions if q.strip() and (q.strip().endswith('?') or q.strip().endswith('.'))]
            all_questions.extend(valid_questions)
            logger.info(f"Extracted {len(valid_questions)} questions from chunk {i}")

        if not all_questions:
            logger.error("No questions were generated from any chunk")
            raise Exception("Failed to generate questions. No valid questions were generated from the input text.")

        # Get top 10 questions
        display_questions = all_questions[:10]
        logger.info(f"Generated {len(all_questions)} questions, displaying top {len(display_questions)}")

        # Initialize answer generation
        try:
            logger.info("Initializing answer generation...")
            answer_llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="gemma2-9b-it",
                temperature=0.1
            )
            logger.info("Answer LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize answer LLM: {str(e)}")
            raise Exception(f"Failed to initialize answer LLM: {str(e)}")

        try:
            logger.info("Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings created successfully")
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")

        try:
            logger.info("Creating vector store...")
            documents = [Document(page_content=chunk) for chunk in chunks]
            vector_store = FAISS.from_documents(documents, embeddings)
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise Exception(f"Failed to create vector store: {str(e)}")

        try:
            logger.info("Creating answer chain...")
            answer_chain = RetrievalQA.from_chain_type(
                llm=answer_llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 1})
            )
            logger.info("Answer chain created successfully")
        except Exception as e:
            logger.error(f"Failed to create answer chain: {str(e)}")
            raise Exception(f"Failed to create answer chain: {str(e)}")

        # Process questions in batches
        qa_list = []
        try:
            logger.info("Processing questions in batches...")
            for i in range(0, len(display_questions), BATCH_SIZE):
                batch = display_questions[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}")
                batch_results = process_batch(batch, answer_chain)
                qa_list.extend(batch_results)
            logger.info(f"Successfully processed {len(qa_list)} questions")
        except Exception as e:
            logger.error(f"Failed to process questions in batches: {str(e)}")
            raise Exception(f"Failed to process questions: {str(e)}")

        # Generate CSV file
        try:
            logger.info("Generating CSV file...")
            output_path = Path("static/output") / f"{Path(file_path).stem}_QA.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                import csv
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Question", "Answer"])
                
                for i in range(0, len(all_questions), BATCH_SIZE):
                    batch = all_questions[i:i + BATCH_SIZE]
                    batch_results = process_batch(batch, answer_chain)
                    for qa in batch_results:
                        csv_writer.writerow([qa["question"], qa["answer"]])
            logger.info(f"CSV file generated successfully at {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV file: {str(e)}")
            raise Exception(f"Failed to generate CSV file: {str(e)}")

        return str(output_path), qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise Exception(f"Document analysis failed: {str(e)}")