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

def file_processing(file_path):
    """Process the input file and return documents for question and answer generation."""
    try:
        # Load data from PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        # Limit pages to prevent memory issues
        if len(data) > 10:
            print(f"Document has {len(data)} pages, limiting to first 10 pages")
            data = data[:10]
        
        question_gen = ''
        for page in data:
            question_gen += page.page_content
            # Limit text length
            if len(question_gen) > 25000:
                print("Text too long, truncating")
                question_gen = question_gen[:25000]
                break

        splitter_ques_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=2000,  # Reduced chunk size for memory efficiency
            chunk_overlap=50  # Reduced overlap
        )
        
        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
        
        # Limit chunks
        if len(chunks_ques_gen) > 5:  # Reduced from 10 to 5
            print(f"Too many chunks ({len(chunks_ques_gen)}), limiting to 5")
            chunks_ques_gen = chunks_ques_gen[:5]
        
        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
        
        splitter_ans_gen = TokenTextSplitter(
            model_name='gpt-3.5-turbo',
            chunk_size=500,  # Further reduced from 1000 to 500
            chunk_overlap=25  # Reduced overlap
        )
        
        document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

        return document_ques_gen, document_answer_gen
    except Exception as e:
        raise Exception(f"Error in file processing: {str(e)}")

def llm_pipeline(file_path):
    """Main LLM pipeline for question and answer generation."""
    try:
        # Check if API key is available
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY is not configured. Please set a valid Groq API key in your environment variables.")
        
        document_ques_gen, document_answer_gen = file_processing(file_path)
        
        # Initialize LLM for question generation
        llm_ques_gen_pipeline = ChatGroq(
            temperature=0.3,
            model_name="gemma2-9b-it"
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

        # Generate questions
        ques_gen_chain = load_summarize_chain(
            llm=llm_ques_gen_pipeline,
            chain_type="refine",
            verbose=True,
            question_prompt=PROMPT_QUESTIONS,
            refine_prompt=REFINE_PROMPT_QUESTIONS
        )
        
        ques = ques_gen_chain.run(document_ques_gen)
        
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
        )
        vector_store = FAISS.from_documents(document_answer_gen, embeddings)
        
        # Initialize LLM for answer generation
        llm_answer_gen = ChatGroq(temperature=0.1, model_name="gemma2-9b-it")
        
        # Process questions
        ques_list = ques.split("\n")
        filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
        
        # Generate 10 questions for CSV
        if len(filtered_ques_list) > 10:
            print(f"Too many questions ({len(filtered_ques_list)}), limiting to 10")
            filtered_ques_list = filtered_ques_list[:10]
        
        # Create answer generation chain
        answer_generation_chain = RetrievalQA.from_chain_type(
            llm=llm_answer_gen,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
            
        return answer_generation_chain, filtered_ques_list
    except Exception as e:
        raise Exception(f"Error in LLM pipeline: {str(e)}") 