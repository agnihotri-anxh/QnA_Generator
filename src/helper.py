from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
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
from typing import List, Tuple, Dict, Optional
import re
import time
from itertools import chain
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
import gc
from datetime import datetime, timedelta
import json

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
RATE_LIMIT_DELAY = 5.0  # Increased base delay
MAX_RETRIES = 5  # Increased retries
BATCH_SIZE = 1  # Reduced batch size to minimize token usage
MAX_TOKENS = 150  # Further reduced max tokens per request
BACKOFF_FACTOR = 2.0  # Exponential backoff factor
MAX_TOKENS_PER_DAY = 500000  # Groq's daily limit
TOKEN_BUFFER = 1000  # Buffer to prevent hitting limit
TOKEN_USAGE_FILE = "token_usage.json"
DAILY_TOKEN_LIMIT = 500000

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
    estimated_tokens = estimate_tokens(content) + estimate_tokens(question) + 100  # Add buffer for system message
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
    
    # Reset counter if it's a new day
    if usage_data["date"] != current_date:
        usage_data = {"date": current_date, "tokens_used": 0}
        save_token_usage(usage_data)
    
    return usage_data["tokens_used"] >= DAILY_TOKEN_LIMIT

def update_token_usage(tokens_used):
    """Update token usage count."""
    usage_data = load_token_usage()
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Reset counter if it's a new day
    if usage_data["date"] != current_date:
        usage_data = {"date": current_date, "tokens_used": 0}
    
    usage_data["tokens_used"] += tokens_used
    save_token_usage(usage_data)

def process_batch(batch, llm, document_content=None):
    results = []
    for question in batch:
        try:
            # Check token limits before processing
            if check_token_limit():
                logger.warning("Daily token limit reached")
                results.append({
                    "question": question,
                    "answer": "Unable to generate answer due to daily token limit. Please try again tomorrow or upgrade your API tier."
                })
                continue

            # Create a more structured prompt for better answer generation
            prompt = f"""Answer this question based on the document:

Document: {document_content if document_content else "No content provided."}

Question: {question}

Answer:"""
            
            # Generate answer using ChatGroq with improved error handling
            try:
                # Create messages using proper LangChain message types
                messages = [
                    SystemMessage(content="You are a document analysis expert. Provide concise, accurate answers based only on the given content."),
                    HumanMessage(content=prompt)
                ]
                
                # Get response from LLM with improved retry mechanism
                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = llm.invoke(messages)
                        if response:  # Ensure we got a valid response
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
                            logger.error(f"Error in LLM invocation: {error_msg}")
                            raise e
                
                if not response:
                    raise ValueError("No response received from LLM after all retries")
                
                # Extract the answer from the response with multiple fallback methods
                answer = None
                if isinstance(response, dict):
                    if 'choices' in response and response['choices']:
                        answer = response['choices'][0]['message']['content'].strip()
                    elif 'content' in response:
                        answer = response['content'].strip()
                elif hasattr(response, 'content'):
                    answer = response.content.strip()
                elif isinstance(response, str):
                    answer = response.strip()
                
                # Validate and clean the answer
                if not answer or answer.isspace():
                    answer = "Unable to generate an answer from the document content."
                elif len(answer) < 10:  # Too short to be a meaningful answer
                    answer = "The generated answer was too short. Please try again."
                
                results.append({
                    "question": question,
                    "answer": answer
                })
                
                # After successful response, update token usage
                if isinstance(response, dict) and 'usage' in response:
                    update_token_usage(response['usage']['total_tokens'])
                else:
                    # Estimate token usage if not provided
                    estimated_tokens = estimate_tokens(prompt) + estimate_tokens(str(response))
                    update_token_usage(estimated_tokens)
                
            except Exception as e:
                logger.error(f"Error in LLM response processing: {str(e)}")
                results.append({
                    "question": question,
                    "answer": "An error occurred while processing your request. Please try again."
                })
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            results.append({
                "question": question,
                "answer": "An error occurred while processing your request. Please try again."
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
            
            # Create proper message format for the chain
            messages = [HumanMessage(content=formatted_prompt)]
            
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

def file_processing(file_path: str) -> Tuple[List[Document], List[Document]]:
    """Process the input file and return documents for question and answer generation."""
    try:
        logger.info(f"Starting file processing for: {file_path}")
        
        # Load data from PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()
        logger.info(f"Document loaded successfully. Number of pages: {len(data)}")

        # Combine all pages for question generation
        question_gen = ''
        for page in data:
            question_gen += page.page_content

        # Split text for question generation
        splitter_ques_gen = TokenTextSplitter(
            chunk_size=10000,
            chunk_overlap=200
        )
        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
        logger.info(f"Text split into {len(document_ques_gen)} chunks for question generation")
        
        # Split text for answer generation
        splitter_ans_gen = TokenTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
        logger.info(f"Text split into {len(document_answer_gen)} chunks for answer generation")

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

def llm_pipeline(file_path: str) -> Tuple[str, List[dict]]:
    """Main pipeline for generating questions and answers."""
    try:
        logger.info("Starting LLM pipeline...")
        
        # Process the document
        document_ques_gen, document_answer_gen = file_processing(file_path)
        
        # Initialize LLM for question generation
        try:
            logger.info("Initializing question generation LLM...")
            llm_ques_gen = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="gemma2-9b-it",
                temperature=0.3,
                max_tokens=MAX_TOKENS
            )
            logger.info("Question generation LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing question generation LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

        # Create prompts for question generation
        try:
            logger.info("Creating question generation prompts...")
            PROMPT_QUESTIONS = PromptTemplate(
                template=prompt_template,
                input_variables=["text"]
            )
            REFINE_PROMPT_QUESTIONS = PromptTemplate(
                input_variables=["existing_answer", "text"],
                template=refine_template
            )
            logger.info("Question generation prompts created successfully")
        except Exception as e:
            logger.error(f"Failed to create question generation prompts: {str(e)}")
            raise Exception(f"Failed to create prompts: {str(e)}")

        # Generate questions
        try:
            logger.info("Generating questions...")
            ques_gen_chain = load_summarize_chain(
                llm=llm_ques_gen,
                chain_type="refine",
                verbose=True,
                question_prompt=PROMPT_QUESTIONS,
                refine_prompt=REFINE_PROMPT_QUESTIONS
            )
            
            for attempt in range(MAX_RETRIES):
                try:
                    questions = ques_gen_chain.run(document_ques_gen)
                    break
                except Exception as e:
                    if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                        if attempt == MAX_RETRIES - 1:
                            raise Exception("Rate limit reached. Please try again later.")
                        backoff_delay = get_backoff_delay(attempt)
                        logger.warning(f"Rate limit hit, retrying in {backoff_delay:.2f} seconds")
                        time.sleep(backoff_delay)
                    else:
                        raise e

            # Process generated questions
            ques_list = questions.split("\n")
            filtered_ques_list = [q.strip() for q in ques_list if q.strip() and (q.strip().endswith('?') or q.strip().endswith('.'))]
            logger.info(f"Generated {len(filtered_ques_list)} questions")
                
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise Exception(f"Failed to generate questions: {str(e)}")

        # Initialize embeddings and vector store
        try:
            logger.info("Initializing embeddings and vector store...")
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(document_answer_gen, embeddings)
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise Exception(f"Failed to create vector store: {str(e)}")

        # Initialize answer generation
        try:
            logger.info("Initializing answer generation...")
            llm_answer_gen = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="gemma2-9b-it",
                temperature=0.1,
                max_tokens=MAX_TOKENS
            )
            
            answer_chain = RetrievalQA.from_chain_type(
                llm=llm_answer_gen,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            logger.info("Answer generation initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing answer generation: {str(e)}")
            raise Exception(f"Failed to initialize answer generation: {str(e)}")

        # Generate answers
        try:
            logger.info("Generating answers...")
            qa_list = []
            for question in filtered_ques_list:
                try:
                    answer = answer_chain.run(question)
                    qa_list.append({
                        "question": question,
                        "answer": answer.strip()
                    })
                except Exception as e:
                    logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
                    qa_list.append({
                        "question": question,
                        "answer": "Error generating answer. Please try again."
                    })
                gc.collect()
            logger.info(f"Generated answers for {len(qa_list)} questions")
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            raise Exception(f"Failed to generate answers: {str(e)}")

        # Generate CSV file
        try:
            logger.info("Generating CSV file...")
            output_file = generate_csv(qa_list, file_path)
            logger.info(f"CSV file generated successfully at {output_file}")
        except Exception as e:
            logger.error(f"Failed to generate CSV file: {str(e)}")
            raise Exception(f"Failed to generate CSV file: {str(e)}")

        return output_file, qa_list

    except Exception as e:
        logger.error(f"Error in LLM pipeline: {str(e)}")
        raise Exception(f"LLM pipeline failed: {str(e)}")