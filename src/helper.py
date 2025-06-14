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
import csv

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('helper.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Rate limiting settings
RATE_LIMIT_DELAY = 1.0  # Increased delay for better stability
MAX_RETRIES = 3
BATCH_SIZE = 3  # Reduced batch size for better stability

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
    try:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error in clean_text: {str(e)}")
        return ""

def process_batch(questions: List[str], answer_chain: RetrievalQA) -> List[dict]:
    """Process a batch of questions together."""
    results = []
    for question in questions:
        try:
            if not question.strip():
                continue
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
            
        # Use smaller chunks for better processing
        splitter = CharacterTextSplitter(
            chunk_size=1000,  # Further reduced chunk size
            chunk_overlap=100, # Reduced chunk overlap
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
                temperature=0.7,
                max_tokens=500  # Further reduced max tokens limit
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", # Using a smaller, efficient model
            model_kwargs={'device': 'cpu'} # Ensure model runs on CPU, not GPU
        )
        logger.info("Embeddings initialized successfully")

        # Create vector store
        logger.info("Creating vector store...")
        vector_store = FAISS.from_texts(chunks, embeddings)
        logger.info("Vector store created successfully")

        # Create retrieval chain
        logger.info("Creating retrieval chain...")
        answer_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(k=2), # Reduced number of retrieved documents
            return_source_documents=False
        )
        logger.info("Retrieval chain created successfully")

        # Generate questions
        logger.info("Generating questions...")
        question_generator_chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=PromptTemplate.from_template(prompt_template),
            verbose=True
        )
        
        # Process chunks to generate initial questions
        initial_questions_str = ""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk_with_retry, chunk, llm, PromptTemplate.from_template(prompt_template)) for chunk in chunks]
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    initial_questions_str += result + "\n"
                    logger.info(f"Processed chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error processing future for chunk {i+1}: {str(e)}")

        # Refine questions if there's a refine template
        qa_list = []
        if refine_template:
            logger.info("Refining questions...")
            refine_chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=PromptTemplate.from_template(refine_template),
                verbose=True
            )
            
            # Apply refine chain to initial questions
            refined_questions = refine_chain.invoke({"input_documents": [Document(page_content=initial_questions_str)], "existing_answer": initial_questions_str, "text": ""}) # Pass initial_questions_str as existing_answer
            
            # Extract and parse questions from the refined output
            if isinstance(refined_questions, dict) and 'output_text' in refined_questions:
                parsed_questions = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', refined_questions['output_text'])
            else:
                parsed_questions = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', str(refined_questions))

            # Process questions in batches for answers
            all_qa_results = []
            for i in range(0, len(parsed_questions), BATCH_SIZE):
                batch = parsed_questions[i:i + BATCH_SIZE]
                all_qa_results.extend(process_batch(batch, answer_chain))
            qa_list = all_qa_results

        if not qa_list:
            raise ValueError("No questions generated or refined.")

        # Save Q&A to CSV
        output_filename = f"qa_{Path(file_path).stem}.csv"
        output_file_path = OUTPUT_DIR / output_filename
        logger.info(f"Saving Q&A to: {output_file_path}")
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["question", "answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qa_list)
        
        logger.info("Q&A saved successfully")
        return str(output_file_path), qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise Exception(f"LLM pipeline failed: {str(e)}")