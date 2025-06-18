from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *
import gc
import psutil

# Groq authentication
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if API key is properly configured
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("WARNING: GROQ_API_KEY environment variable is not set or is invalid.")
    print("Please set a valid Groq API key in your environment variables.")
    print("The application will start but Q&A generation will fail.")
    # Don't raise an exception here to allow the app to start
    GROQ_API_KEY = None
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def check_memory():
    """Check available memory and return True if sufficient"""
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(f"Available memory: {available_gb:.2f} GB")
        return available_gb > 0.5  # Need at least 500MB
    except:
        return True  # If we can't check, assume it's OK

def force_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    gc.collect()
    gc.collect()

def file_processing(file_path):
    """Process the input file and return documents for question and answer generation."""
    try:
        # Check memory before processing
        if not check_memory():
            raise Exception("Insufficient memory available for processing")
        
        # Load data from PDF with strict limits
        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        # Very aggressive page limiting for memory-constrained environments
        if len(data) > 4:  # Reduced from 8 to 4
            print(f"Document has {len(data)} pages, limiting to first 4 pages")
            data = data[:4]
        
        question_gen = ''
        for page in data:
            question_gen += page.page_content
            # Very strict text length limit
            if len(question_gen) > 15000:  # Reduced from 20000 to 15000
                print("Text too long, truncating")
                question_gen = question_gen[:15000]
                break

        # Force cleanup after text extraction
        force_cleanup()

        splitter_ques_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=1000,  # Reduced from 1500 to 1000
            chunk_overlap=20  # Reduced overlap
        )
        
        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
        
        # Very strict chunk limiting
        if len(chunks_ques_gen) > 3:  # Reduced from 4 to 3
            print(f"Too many chunks ({len(chunks_ques_gen)}), limiting to 3")
            chunks_ques_gen = chunks_ques_gen[:3]
        
        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
        
        splitter_ans_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=300,  # Reduced from 400 to 300
            chunk_overlap=15  # Reduced overlap
        )
        
        document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

        # Force cleanup
        force_cleanup()

        return document_ques_gen, document_answer_gen
    except Exception as e:
        raise Exception(f"Error in file processing: {str(e)}")

def llm_pipeline(file_path):
    """Main LLM pipeline for question and answer generation."""
    try:
        # Check if API key is available
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY is not configured. Please set a valid Groq API key in your environment variables.")
        
        # Check memory before starting
        if not check_memory():
            raise Exception("Insufficient memory available for LLM processing")
        
        document_ques_gen, document_answer_gen = file_processing(file_path)
        
        # Initialize LLM for question generation with memory optimization
        llm_ques_gen_pipeline = ChatGroq(
            temperature=0.3,
            model_name="gemma2-9b-it",
            max_tokens=500  # Limit token usage
        )

        # Create prompts
        PROMPT_QUESTIONS = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
    
        REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )

        # Generate questions with memory monitoring
        ques_gen_chain = load_summarize_chain(
            llm=llm_ques_gen_pipeline,
            chain_type="refine",
            verbose=False,  # Reduce logging overhead
            question_prompt=PROMPT_QUESTIONS,
            refine_prompt=REFINE_PROMPT_QUESTIONS
        )
        
        # Use invoke() instead of run() to fix deprecation warning
        ques = ques_gen_chain.invoke({"input_documents": document_ques_gen})
        if isinstance(ques, dict) and 'output_text' in ques:
            ques = ques['output_text']
        
        # Force cleanup after question generation
        force_cleanup()
        
        # Check memory before embeddings
        if not check_memory():
            raise Exception("Insufficient memory for embedding generation")
        
        # Initialize embeddings with memory optimization
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store with limited documents
        if len(document_answer_gen) > 10:  # Limit vector store size
            document_answer_gen = document_answer_gen[:10]
        
        vector_store = FAISS.from_documents(document_answer_gen, embeddings)
        
        # Force cleanup after vector store creation
        force_cleanup()
        
        # Initialize LLM for answer generation
        llm_answer_gen = ChatGroq(
            temperature=0.1, 
            model_name="gemma2-9b-it",
            max_tokens=300  # Limit token usage
        )
        
        # Process questions with strict limits
        ques_list = ques.split("\n")
        filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
        
        # Very strict question limiting
        if len(filtered_ques_list) > 5:  # Reduced from 8 to 5
            print(f"Too many questions ({len(filtered_ques_list)}), limiting to 5")
            filtered_ques_list = filtered_ques_list[:5]
        
        # Create answer generation chain
        answer_generation_chain = RetrievalQA.from_chain_type(
            llm=llm_answer_gen,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2})  # Limit retrieved documents
        )
        
        # Force cleanup
        force_cleanup()
            
        return answer_generation_chain, filtered_ques_list
    except Exception as e:
        raise Exception(f"Error in LLM pipeline: {str(e)}") 