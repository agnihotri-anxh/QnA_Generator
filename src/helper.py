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
RATE_LIMIT_DELAY = 2.0  # Increased base delay
MAX_RETRIES = 3
BATCH_SIZE = 2  # Reduced batch size
MAX_TOKENS = 300  # Reduced max tokens per request
BACKOFF_FACTOR = 2.0  # Exponential backoff factor

def get_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay."""
    return RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** attempt)

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

def process_batch(batch, llm, document_content=None):
    results = []
    for question in batch:
        try:
            # Create a more structured prompt for better answer generation
            prompt = f"""You are an expert in document analysis. Your task is to answer the following question based on the provided document content.

Document Content:
{document_content if document_content else "No document content provided."}

Question: {question}

Instructions:
1. Provide a clear and concise answer based on the document content
2. If the answer is not in the content, explicitly state "The answer is not available in the provided document content"
3. Focus on accuracy and relevance
4. Keep the answer concise but informative

Answer:"""
            
            # Generate answer using ChatGroq with improved error handling
            try:
                # Create a message for the LLM with system context
                messages = [
                    {"role": "system", "content": "You are an expert in document analysis and question answering. Your responses should be accurate, concise, and based solely on the provided document content."},
                    {"role": "user", "content": prompt}
                ]
                
                # Get response from LLM with improved retry mechanism
                for attempt in range(MAX_RETRIES):
                    try:
                        response = llm.invoke(messages)
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                            if attempt == MAX_RETRIES - 1:
                                raise e
                            backoff_delay = get_backoff_delay(attempt)
                            logger.warning(f"Rate limit hit, retrying in {backoff_delay:.2f} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
                            time.sleep(backoff_delay)
                        else:
                            raise e
                
                # Extract the answer from the response with multiple fallback methods
                if isinstance(response, dict) and 'choices' in response:
                    answer = response['choices'][0]['message']['content'].strip()
                elif hasattr(response, 'content'):
                    answer = response.content.strip()
                elif isinstance(response, str):
                    answer = response.strip()
                else:
                    answer = str(response).strip()
                
                # Validate and clean the answer
                if not answer or answer.isspace():
                    answer = "Unable to generate an answer from the document content."
                elif len(answer) < 10:  # Too short to be a meaningful answer
                    answer = "The generated answer was too short. Please try again."
                
                results.append({
                    "question": question,
                    "answer": answer
                })
                
            except Exception as e:
                logger.error(f"Error in LLM response processing: {str(e)}")
                # Try alternative approach with simpler prompt
                try:
                    simple_prompt = f"Based on this content: {document_content}\n\nQuestion: {question}\n\nAnswer:"
                    response = llm.invoke(simple_prompt)
                    answer = str(response).strip()
                    
                    if answer and not answer.isspace() and len(answer) >= 10:
                        results.append({
                            "question": question,
                            "answer": answer
                        })
                    else:
                        results.append({
                            "question": question,
                            "answer": "Unable to generate a valid answer from the document content."
                        })
                except Exception as e2:
                    logger.error(f"Error in alternative LLM response processing: {str(e2)}")
                    results.append({
                        "question": question,
                        "answer": "Error processing LLM response. Please try again."
                    })
                
        except Exception as e:
            logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
            results.append({
                "question": question,
                "answer": "Error generating answer. Please try again."
            })
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
        
        # Load document in smaller batches
        logger.info("Loading document...")
        data = []
        batch_size = 5  # Process 5 pages at a time
        
        # Load and process in batches
        for i in range(0, len(loader.load()), batch_size):
            batch = loader.load()[i:i + batch_size]
            processed_batch = []
            
            for page in batch:
                # Clean and process each page
                cleaned_text = clean_text(page.page_content)
                if cleaned_text.strip():
                    processed_batch.append(cleaned_text)
            
            # Combine batch results
            data.extend(processed_batch)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        if not data:
            raise ValueError("No content extracted from document")
            
        logger.info(f"Document loaded successfully. Number of sections: {len(data)}")
        
        # Use smaller chunks for better processing
        splitter = CharacterTextSplitter(
            chunk_size=1000,  # Reduced chunk size
            chunk_overlap=100,  # Reduced overlap
            separator="\n"
        )

        # Process chunks in batches
        all_chunks = []
        for section in data:
            section_chunks = splitter.split_text(section)
            all_chunks.extend(section_chunks)
            # Force garbage collection after each section
            gc.collect()
        
        logger.info(f"Text split into {len(all_chunks)} chunks")
        
        if not all_chunks:
            raise ValueError("No chunks generated after text splitting")

        return all_chunks
    
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        raise Exception(f"Document analysis failed: {str(e)}")

def llm_pipeline(chunks: List[str], file_path: str) -> Tuple[str, List[dict]]:
    try:
        logger.info("Starting LLM pipeline...")
        
        # Process chunks in smaller batches to reduce memory usage
        BATCH_SIZE = 2  # Reduced batch size
        
        # Combine chunks for document content
        document_content = "\n".join(chunks)
        logger.info(f"Document content length: {len(document_content)} characters")
        
        # Initialize LLM with optimized settings
        try:
            logger.info("Initializing LLM...")
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY environment variable is not set")
                
            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="gemma2-9b-it",
                temperature=0.7,
                max_tokens=MAX_TOKENS  # Use the new token limit
            )
            # Test the LLM connection
            test_response = llm.invoke([{"role": "user", "content": "Test connection"}])
            logger.info("LLM initialized and tested successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM. Please check your GROQ_API_KEY and internet connection: {str(e)}")

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

        # Process chunks in smaller batches
        all_questions = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:  # Reduced workers
                    futures = [executor.submit(process_chunk_with_retry, chunk, llm, prompt) for chunk in batch]
                    results = [f.result() for f in futures]
                
                for result in results:
                    if result:
                        questions = result.split('\n')
                        valid_questions = [q.strip() for q in questions if q.strip() and (q.strip().endswith('?') or q.strip().endswith('.'))]
                        all_questions.extend(valid_questions)
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                continue

        if not all_questions:
            logger.error("No questions were generated from any chunk")
            raise Exception("Failed to generate questions. No valid questions were generated from the input text.")

        # Get top 5 questions instead of 10 to reduce memory usage
        display_questions = all_questions[:5]
        logger.info(f"Generated {len(all_questions)} questions, displaying top {len(display_questions)}")

        # Initialize answer generation with reduced memory usage
        try:
            logger.info("Initializing answer generation...")
            answer_llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="gemma2-9b-it",
                temperature=0.1,
                max_tokens=MAX_TOKENS  # Use the new token limit
            )
            logger.info("Answer LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize answer LLM: {str(e)}")
            raise Exception(f"Failed to initialize answer LLM: {str(e)}")

        # Process questions in smaller batches
        qa_list = []
        try:
            logger.info("Processing questions in batches...")
            for i in range(0, len(display_questions), BATCH_SIZE):
                batch = display_questions[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}")
                batch_results = process_batch(batch, answer_llm, document_content)
                qa_list.extend(batch_results)
                # Force garbage collection after each batch
                gc.collect()
            logger.info(f"Successfully processed {len(qa_list)} questions")
        except Exception as e:
            logger.error(f"Failed to process questions in batches: {str(e)}")
            raise Exception(f"Failed to process questions: {str(e)}")

        # Generate CSV file with reduced memory usage
        try:
            logger.info("Generating CSV file...")
            output_path = Path("static/output") / f"{Path(file_path).stem}_QA.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                import csv
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Question", "Answer"])
                
                for i in range(0, len(all_questions), BATCH_SIZE):
                    batch = all_questions[i:i + BATCH_SIZE]
                    batch_results = process_batch(batch, answer_llm, document_content)
                    for qa in batch_results:
                        csv_writer.writerow([qa["question"], qa["answer"]])
                    # Force garbage collection after each batch
                    gc.collect()
            logger.info(f"CSV file generated successfully at {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV file: {str(e)}")
            raise Exception(f"Failed to generate CSV file: {str(e)}")

        return str(output_path), qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise Exception(f"LLM pipeline failed: {str(e)}")