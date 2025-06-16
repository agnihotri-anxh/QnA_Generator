from fastapi import FastAPI, Form, Request, Response, File, HTTPException, status, UploadFile
from fastapi.responses import RedirectResponse, FileResponse
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

# Create necessary directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = STATIC_DIR / "docs"
OUTPUT_DIR = STATIC_DIR / "output"

for directory in [DOCS_DIR, OUTPUT_DIR]:
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
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None)
):
    try:
        logger.debug(f"Received file upload request: {file.filename}")
        
        # Use provided filename or original filename
        save_filename = filename if filename else file.filename
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt'}
        file_ext = os.path.splitext(save_filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload a PDF, Word, or PowerPoint file."
            )

        # Save the file
        pdf_filename = os.path.join(DOCS_DIR, save_filename)
        async with aiofiles.open(pdf_filename, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"File saved successfully: {pdf_filename}")
        
        return {
            "msg": "success",
            "pdf_filename": pdf_filename
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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
async def analyze_document(request: Request, pdf_filename: str = Form(...)):
    try:
        logger.info(f"Starting document analysis for: {pdf_filename}")
        
        # Validate file exists
        if not os.path.exists(pdf_filename):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Generate CSV file and get content
        output_file, qa_list, content_sections = get_csv(pdf_filename)
        
        return {
            "output_file": output_file,
            "qa_list": qa_list,
            "document_content": content_sections
        }
        
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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
    uvicorn.run(
        "app:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost
        port=8080,
        reload=True,
        log_level="debug"
    )