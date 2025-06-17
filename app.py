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
from src.helper import llm_pipeline, run_with_timeout
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

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the app is working."""
    return {"status": "ok", "message": "QnA Generator is running!", "timestamp": time.time()}

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
        logger.info(f"Starting CSV generation with timeout for file: {file_path}")
        
        # Use timeout wrapper for the LLM pipeline
        from src.helper import run_with_timeout, llm_pipeline
        
        try:
            output_file, qa_list = run_with_timeout(llm_pipeline, [file_path], timeout_seconds=300)
            logger.info(f"LLM pipeline completed successfully. Output: {output_file}, Q&A count: {len(qa_list)}")
            return output_file, qa_list
        except TimeoutError as e:
            logger.error(f"LLM pipeline timed out: {str(e)}")
            raise HTTPException(
                status_code=408,
                detail="Document processing timed out. Please try with a smaller document or try again later."
            )
        except Exception as e:
            logger.error(f"LLM pipeline failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_csv: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    try:
        logger.info(f"Starting analysis for file: {pdf_filename}")
        
        # Validate file exists
        if not os.path.exists(pdf_filename):
            logger.error(f"File not found: {pdf_filename}")
            raise HTTPException(
                status_code=404,
                detail=f"File {pdf_filename} not found"
            )
        
        logger.info("Calling get_csv function...")
        output_file, qa_data = get_csv(pdf_filename)
        logger.info(f"Analysis completed. Output file: {output_file}, Q&A count: {len(qa_data)}")
        
        response_data = jsonable_encoder(json.dumps({
            "output_file": output_file,
            "qa_data": qa_data
        }))
        res = Response(response_data)
        return res
        
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a more specific error message
        error_detail = f"Error analyzing document: {str(e)}"
        if "memory" in str(e).lower():
            error_detail = "Document processing failed due to memory constraints. Please try with a smaller document."
        elif "timeout" in str(e).lower():
            error_detail = "Document processing timed out. Please try again."
        
        raise HTTPException(
            status_code=500,
            detail=error_detail
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