from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
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
from typing import List, Tuple, Dict, Optional
import re
import time
from itertools import chain
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
import gc
from datetime import datetime, timedelta
import json
import signal
import functools
import psutil
import sys

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

# Memory and performance settings
MAX_DOCUMENT_SIZE_MB = 10  # Maximum document size in MB
MAX_CHUNK_SIZE = 2000  # Reduced chunk size for memory efficiency (was 5000)
MAX_CHUNKS = 20  # Maximum number of chunks to process
MAX_QUESTIONS = 10  # Maximum number of questions to generate
RATE_LIMIT_DELAY = 5.0
MAX_RETRIES = 3  # Reduced retries
BATCH_SIZE = 1
MAX_TOKENS = 100  # Further reduced max tokens
BACKOFF_FACTOR = 2.0
MAX_TOKENS_PER_DAY = 500000
TOKEN_BUFFER = 1000
TOKEN_USAGE_FILE = "token_usage.json"
DAILY_TOKEN_LIMIT = 500000

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_limit():
    """Check if memory usage is approaching limits."""
    memory_usage = get_memory_usage()
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")
    return memory_usage > 400  # 400MB limit for Render's 512MB

def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()
    memory_usage = get_memory_usage()
    logger.info(f"Memory after GC: {memory_usage:.2f} MB")

def get_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay."""
    return RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** attempt)

def get_document_loader(file_path):
    logger.info(f"Getting document loader for file: {file_path}")
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_DOCUMENT_SIZE_MB:
            raise ValueError(f"File too large ({file_size_mb:.2f} MB). Maximum allowed: {MAX_DOCUMENT_SIZE_MB} MB")
            
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

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error estimating tokens: {str(e)}")
        return len(text.split()) * 1.3  # Rough estimate as fallback

def check_token_limit(content: str, question: str) -> bool:
    """Check if the request would exceed token limits."""
    estimated_tokens = estimate_tokens(content) + estimate_tokens(question) + 100
    return estimated_tokens <= (MAX_TOKENS - TOKEN_BUFFER)

def load_token_usage():
    """Load token usage from file."""
    try:
        if os.path.exists(TOKEN_USAGE_FILE):
            with open(TOKEN_USAGE_FILE, 'r') as f:
                return json.load(f)
        return {"date": datetime.now().strftime("%Y-%m-%d"), "tokens_used": 0}
    except Exception as e:
        logger.error(f"Error loading token usage: {str(e)}")
        return {"date": datetime.now().strftime("%Y-%m-%d"), "tokens_used": 0}

def save_token_usage(usage_data):
    """Save token usage to file."""
    try:
        with open(TOKEN_USAGE_FILE, 'w') as f:
            json.dump(usage_data, f)
    except Exception as e:
        logger.error(f"Error saving token usage: {str(e)}")

def check_token_limit():
    """Check if we've hit the daily token limit."""
    usage_data = load_token_usage()
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if usage_data["date"] != current_date:
        usage_data = {"date": current_date, "tokens_used": 0}
        save_token_usage(usage_data)
    
    return usage_data["tokens_used"] >= DAILY_TOKEN_LIMIT

def update_token_usage(tokens_used):
    """Update token usage count."""
    usage_data = load_token_usage()
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if usage_data["date"] != current_date:
        usage_data = {"date": current_date, "tokens_used": 0}
    
    usage_data["tokens_used"] += tokens_used
    save_token_usage(usage_data)

def process_batch(batch, llm, document_content=None):
    results = []
    for question in batch:
        try:
            if check_token_limit():
                logger.warning("Daily token limit reached")
                results.append({
                    "question": question,
                    "answer": "Unable to generate answer due to daily token limit. Please try again tomorrow or upgrade your API tier."
                })
                continue

            prompt = f"""Answer this question based on the document:

Document: {document_content if document_content else "No content provided."}

Question: {question}

Answer:"""
            
            try:
                messages = [
                    SystemMessage(content="You are a document analysis expert. Provide concise, accurate answers based only on the given content."),
                    HumanMessage(content=prompt)
                ]
                
                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = llm.invoke(messages)
                        if response:
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                            if attempt == MAX_RETRIES - 1:
                                results.append({
                                    "question": question,
                                    "answer": "Unable to generate answer due to API rate limit. Please try again later or upgrade your API tier."
                                })
                                logger.warning(f"Rate limit reached after {MAX_RETRIES} attempts. Skipping remaining questions.")
                                return results
                            backoff_delay = get_backoff_delay(attempt)
                            logger.warning(f"Rate limit hit, retrying in {backoff_delay:.2f} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
                            time.sleep(backoff_delay)
                        else:
                            raise e

                if response:
                    results.append({
                        "question": question,
                        "answer": response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    })
                else:
                    results.append({
                        "question": question,
                        "answer": "Unable to generate answer. Please try again."
                    })

            except Exception as e:
                logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
                results.append({
                    "question": question,
                    "answer": "Error generating answer. Please try again."
                })

        except Exception as e:
            logger.error(f"Error processing question: {question}. Error: {str(e)}")
            results.append({
                "question": question,
                "answer": "Error processing question. Please try again."
            })

    return results

def process_chunk_with_retry(chunk: str, llm: ChatGroq, prompt: PromptTemplate, max_retries: int = MAX_RETRIES) -> str:
    """Process a chunk with retry mechanism and memory management."""
    for attempt in range(max_retries):
        try:
            if check_memory_limit():
                logger.warning("Memory limit approaching, forcing garbage collection")
                force_garbage_collection()
            
            response = llm.invoke(prompt.format(text=chunk))
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                if attempt == max_retries - 1:
                    raise Exception("Rate limit reached. Please try again later.")
                backoff_delay = get_backoff_delay(attempt)
                logger.warning(f"Rate limit hit, retrying in {backoff_delay:.2f} seconds")
                time.sleep(backoff_delay)
            else:
                raise e

def file_processing(file_path: str) -> Tuple[List[Document], List[Document]]:
    """Process the input file with memory optimization."""
    try:
        logger.info(f"Starting file processing for: {file_path}")
        
        # Check memory before processing
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Load data from PDF with size limit
        loader = get_document_loader(file_path)
        data = loader.load()
        logger.info(f"Document loaded successfully. Number of pages: {len(data)}")

        # Limit the number of pages processed
        if len(data) > 20:  # Limit to first 20 pages
            logger.warning(f"Document has {len(data)} pages, limiting to first 20 pages")
            data = data[:20]

        # Combine all pages for question generation
        question_gen = ''
        for page in data:
            question_gen += page.page_content
            if len(question_gen) > 50000:  # Limit text size
                logger.warning("Text too long, truncating for question generation")
                question_gen = question_gen[:50000]
                break

        # Split text for question generation with smaller chunks
        splitter_ques_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=100  # Reduced overlap
        )
        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
        
        # Limit number of chunks
        if len(chunks_ques_gen) > MAX_CHUNKS:
            logger.warning(f"Too many chunks ({len(chunks_ques_gen)}), limiting to {MAX_CHUNKS}")
            chunks_ques_gen = chunks_ques_gen[:MAX_CHUNKS]
        
        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
        logger.info(f"Text split into {len(document_ques_gen)} chunks for question generation")
        
        # Split text for answer generation with even smaller chunks
        splitter_ans_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=500,  # Smaller chunks for answer generation (was 1000)
            chunk_overlap=50
        )
        document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
        logger.info(f"Text split into {len(document_answer_gen)} chunks for answer generation")

        # Force garbage collection after processing
        force_garbage_collection()
        
        return document_ques_gen, document_answer_gen
    
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        raise Exception(f"Document analysis failed: {str(e)}")

def generate_csv(qa_list: List[dict], file_path: str) -> str:
    """Generate a CSV file from the Q&A list."""
    try:
        output_path = Path("static/output") / f"{Path(file_path).stem}_QA.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            import csv
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Question", "Answer"])
            for qa in qa_list:
                csv_writer.writerow([qa["question"], qa["answer"]])
        return str(output_path)
    except Exception as e:
        logger.error(f"Error generating CSV file: {str(e)}")
        raise Exception(f"Failed to generate CSV file: {str(e)}")

def initialize_embeddings():
    """Initialize embeddings with memory optimization."""
    try:
        logger.info("Initializing embeddings with lightweight model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Lighter model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 2}  # Smaller batch size
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise

def create_vector_store(documents, embeddings):
    """Create vector store with memory optimization."""
    max_retries = 2  # Reduced retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Creating vector store (attempt {attempt + 1}/{max_retries})...")
            
            # Limit documents for vector store
            if len(documents) > 10:
                logger.warning(f"Too many documents ({len(documents)}), limiting to 10 for vector store")
                documents = documents[:10]
            
            vector_store = FAISS.from_documents(documents, embeddings)
            logger.info("Vector store created successfully")
            return vector_store
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to create vector store after {max_retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def timeout_decorator(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@timeout_decorator(180)  # Reduced timeout to 3 minutes
def llm_pipeline(file_path: str) -> Tuple[str, List[dict]]:
    """Main LLM pipeline with memory optimization."""
    try:
        logger.info("Starting LLM pipeline...")
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Process file
        logger.info(f"Starting file processing for: {file_path}")
        document_ques_gen, document_answer_gen = file_processing(file_path)
        
        # Check memory after file processing
        memory_after_processing = get_memory_usage()
        logger.info(f"Memory after file processing: {memory_after_processing:.2f} MB")
        
        # Skip embeddings if memory usage is high
        use_embeddings = False
        if memory_after_processing < 300:  # Only use embeddings if memory is low
            try:
                embeddings = initialize_embeddings()
                use_embeddings = True
                logger.info("Embeddings initialized successfully")
                
                try:
                    vector_store = create_vector_store(document_ques_gen, embeddings)
                    del embeddings
                    force_garbage_collection()
                except Exception as e:
                    logger.warning(f"Failed to create vector store, using fallback mode: {str(e)}")
                    use_embeddings = False
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings, using fallback mode: {str(e)}")
                use_embeddings = False
        
        # Initialize single LLM instance for both questions and answers
        try:
            logger.info("Initializing LLM...")
            llm = ChatGroq(
                temperature=0.3,
                model_name="mistral-saba-24b"
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

        # Generate questions with memory monitoring
        try:
            logger.info("Generating questions...")
            
            # Use a simpler approach for question generation
            questions = []
            for i, chunk in enumerate(document_ques_gen[:5]):  # Limit to first 5 chunks
                if check_memory_limit():
                    logger.warning("Memory limit reached during question generation")
                    break
                
                try:
                    prompt = f"""Generate 2-3 questions based on this text: {chunk.page_content[:2000]}"""
                    response = llm.invoke(prompt)
                    if response:
                        content = response.content if hasattr(response, 'content') else str(response)
                        questions.extend([q.strip() for q in content.split('\n') if q.strip() and '?' in q])
                except Exception as e:
                    logger.error(f"Error generating questions from chunk {i}: {str(e)}")
                    continue
                
                if len(questions) >= MAX_QUESTIONS:
                    break
            
            # Limit number of questions
            if len(questions) > MAX_QUESTIONS:
                questions = questions[:MAX_QUESTIONS]
            
            logger.info(f"Generated {len(questions)} questions")
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            # Fallback: generate simple questions
            questions = ["What is the main topic of this document?", "What are the key points discussed?"]

        # Generate answers
        try:
            logger.info("Generating answers...")
            qa_list = []
            
            for i, question in enumerate(questions):
                if check_memory_limit():
                    logger.warning("Memory limit reached during answer generation")
                    break
                
                try:
                    if use_embeddings:
                        answer_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vector_store.as_retriever()
                        )
                        answer = answer_chain.run(question)
                    else:
                        # Simple answer generation without embeddings
                        prompt = f"Answer this question based on the document content: {question}"
                        response = llm.invoke(prompt)
                        answer = response.content if hasattr(response, 'content') else str(response)
                    
                    qa_list.append({
                        "question": question,
                        "answer": answer.strip()
                    })
                    
                    # Force garbage collection every few questions
                    if (i + 1) % 3 == 0:
                        force_garbage_collection()
                        
                except Exception as e:
                    logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
                    qa_list.append({
                        "question": question,
                        "answer": "Error generating answer. Please try again."
                    })
            
            logger.info(f"Generated answers for {len(qa_list)} questions")
            
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            raise Exception(f"Failed to generate answers: {str(e)}")

        # Clean up
        if use_embeddings:
            del vector_store
        del llm
        force_garbage_collection()

        # Generate CSV file
        try:
            logger.info("Generating CSV file...")
            output_file = generate_csv(qa_list, file_path)
            logger.info(f"CSV file generated successfully at {output_file}")
        except Exception as e:
            logger.error(f"Failed to generate CSV file: {str(e)}")
            raise Exception(f"Failed to generate CSV file: {str(e)}")

        final_memory = get_memory_usage()
        logger.info(f"Final memory usage: {final_memory:.2f} MB")

        return output_file, qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise Exception(f"LLM pipeline failed: {str(e)}")