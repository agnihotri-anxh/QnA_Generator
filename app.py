from fastapi import FastAPI, Form, Request, Response, File, HTTPException, status, UploadFile
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline
import logging
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)  # Enable debug mode

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = STATIC_DIR / "docs"
OUTPUT_DIR = STATIC_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"

for directory in [DOCS_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

def get_document_loader(file_path: str):
    """Get the appropriate document loader based on file extension."""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        return PyPDFLoader(file_path)
    elif file_ext in ['.docx', '.doc']:
        return UnstructuredWordDocumentLoader(file_path)
    elif file_ext in ['.pptx', '.ppt']:
        return UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

@app.get("/")
async def index(request: Request):
    logger.debug("Accessing index page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and DOCX files are allowed"
            )
        
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return JSONResponse(content={
            "msg": "success",
            "pdf_filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

def get_csv(file_path: str) -> tuple[str, list[dict], list[str]]:
    """Generate CSV file from Q&A pairs and return document content."""
    try:
        logger.info(f"Starting CSV generation for file: {file_path}")
        
        # Get Q&A pairs from LLM pipeline
        output_file, qa_list = llm_pipeline(file_path)
        
        # Extract document content
        loader = get_document_loader(file_path)
        documents = loader.load()
        content_sections = []
        
        for doc in documents:
            # Split content into sections (e.g., by paragraphs)
            sections = [section.strip() for section in doc.page_content.split('\n\n') if section.strip()]
            content_sections.extend(sections)
        
        # Generate CSV file
        output_path = OUTPUT_DIR / "QA.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Question", "Answer"])
            
            for qa in qa_list:
                csv_writer.writerow([qa["question"], qa["answer"]])
        
        logger.info(f"CSV file generated successfully: {output_path}")
        return str(output_path), qa_list, content_sections
        
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating CSV: {str(e)}"
        )

@app.post("/analyze")
async def analyze_document(pdf_filename: str = Form(...)):
    try:
        # Validate file exists
        file_path = UPLOAD_DIR / pdf_filename
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File {pdf_filename} not found"
            )

        # Generate CSV and get content
        csv_filename = generate_csv(str(file_path))
        content = get_document_content(str(file_path))
        
        # Generate Q&A pairs
        qa_list = generate_qa_pairs(content)
        
        # Ensure all data is JSON serializable
        response_data = {
            "qa_list": [
                {
                    "question": str(qa["question"]),
                    "answer": str(qa["answer"])
                }
                for qa in qa_list
            ],
            "document_content": [str(section) for section in content]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing document: {str(e)}"
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        return FileResponse(file_path, filename=filename)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    logger.info("Starting server...")
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    logger.info(f"Server will be available at http://{host}:{port}")
    
    # Ensure directories exist
    for directory in [DOCS_DIR, OUTPUT_DIR, UPLOAD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )