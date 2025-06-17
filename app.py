from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status, UploadFile
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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment variable or use default
PORT = int(os.getenv("PORT", 10000))
logger.info(f"Starting server on port {PORT}")

app = FastAPI()

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
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"message": "pong"}

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    try:
        base_folder = 'static/docs/'
        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)
        pdf_filename = os.path.join(base_folder, filename)

        async with aiofiles.open(pdf_filename, 'wb') as f:
            await f.write(pdf_file)
     
        response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
        res = Response(response_data)
        return res
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

def get_csv(file_path):
    try:
        output_file, qa_list = llm_pipeline(file_path)
        
        # The qa_list already contains the questions and answers
        # We just need to create the CSV file
        base_folder = 'static/output/'
        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)
        
        # Use the output file path returned by llm_pipeline
        csv_file_path = output_file
        
        # Return the file path and qa_data for frontend
        return csv_file_path, qa_list
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating CSV: {str(e)}"
        )

@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    try:
        output_file, qa_data = get_csv(pdf_filename)
        response_data = jsonable_encoder(json.dumps({
            "output_file": output_file,
            "qa_data": qa_data
        }))
        res = Response(response_data)
        return res
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

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("Application starting up...")
    # Ensure directories exist
    for directory in [DOCS_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")
    logger.info("Application startup complete!")

if __name__ == "__main__":
    logger.info("Starting server...")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Server will be available at http://{host}:{port}")
    
    # Ensure directories exist
    for directory in [DOCS_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )