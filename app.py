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
from src.helper import llm_pipeline, get_memory_usage, check_memory_limit, force_garbage_collection
import logging
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from fastapi.middleware.cors import CORSMiddleware
import time
import psutil
import asyncio

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
    try:
        memory_usage = get_memory_usage()
        return {
            "status": "healthy", 
            "timestamp": time.time(),
            "memory_usage_mb": round(memory_usage, 2)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"message": "pong"}

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    try:
        # Check memory before upload
        initial_memory = get_memory_usage()
        logger.info(f"Memory before upload: {initial_memory:.2f} MB")
        
        # Check file size
        file_size_mb = len(pdf_file) / (1024 * 1024)
        if file_size_mb > 5:  # Reduced from 10MB to 5MB
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.2f} MB). Maximum allowed: 5 MB"
            )
        
        base_folder = 'static/docs/'
        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)
        pdf_filename = os.path.join(base_folder, filename)

        async with aiofiles.open(pdf_filename, 'wb') as f:
            await f.write(pdf_file)
        
        # Force garbage collection after upload
        force_garbage_collection()
        
        response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
        res = Response(response_data)
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

def get_csv(file_path):
    try:
        # Check memory before processing
        initial_memory = get_memory_usage()
        logger.info(f"Memory before CSV generation: {initial_memory:.2f} MB")
        
        if check_memory_limit():
            logger.warning("Memory limit reached before processing")
            raise Exception("Server is currently under high load. Please try again later.")
        
        output_file, qa_list = llm_pipeline(file_path)
        
        # Force garbage collection after processing
        force_garbage_collection()
        
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
        logger.info(f"Starting analysis for file: {pdf_filename}")
        
        # Check memory before analysis
        initial_memory = get_memory_usage()
        logger.info(f"Memory before analysis: {initial_memory:.2f} MB")
        
        # Validate file exists
        if not os.path.exists(pdf_filename):
            logger.error(f"File not found: {pdf_filename}")
            raise HTTPException(
                status_code=404,
                detail=f"File {pdf_filename} not found"
            )
        
        # Check file size
        file_size_mb = os.path.getsize(pdf_filename) / (1024 * 1024)
        if file_size_mb > 5:  # Reduced from 10MB to 5MB
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.2f} MB). Maximum allowed: 5 MB"
            )
        
        # Check memory limit
        if check_memory_limit():
            logger.warning("Memory limit reached before analysis")
            raise HTTPException(
                status_code=503,
                detail="Server is currently under high load. Please try again later."
            )
        
        # Add timeout wrapper for the analysis
        try:
            logger.info("Calling get_csv function...")
            # Run the analysis with a timeout
            loop = asyncio.get_event_loop()
            output_file, qa_data = await loop.run_in_executor(None, get_csv, pdf_filename)
            logger.info(f"Analysis completed. Output file: {output_file}, Q&A count: {len(qa_data)}")
        except asyncio.TimeoutError:
            logger.error("Analysis timed out")
            raise HTTPException(
                status_code=408,
                detail="Analysis timed out. Please try with a smaller document."
            )
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
        
        # Check final memory usage
        final_memory = get_memory_usage()
        logger.info(f"Memory after analysis: {final_memory:.2f} MB")
        
        response_data = jsonable_encoder(json.dumps({
            "output_file": output_file,
            "qa_data": qa_data
        }))
        res = Response(response_data)
        return res
        
    except HTTPException:
        raise
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
        elif "rate_limit" in str(e).lower():
            error_detail = "API rate limit reached. Please try again later."
        
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

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status and memory usage."""
    try:
        memory_usage = get_memory_usage()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        return {
            "status": "debug_info",
            "memory_usage_mb": round(memory_usage, 2),
            "cpu_percent": cpu_percent,
            "disk_free_gb": round(disk_usage.free / (1024**3), 2),
            "disk_total_gb": round(disk_usage.total / (1024**3), 2),
            "memory_limit_reached": check_memory_limit(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("Application starting up...")
    # Ensure directories exist
    for directory in [DOCS_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")
    
    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    logger.info("Application startup complete!")

if __name__ == "__main__":
    logger.info("Starting server...")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Server will be available at http://{host}:{port}")
    
    # Ensure directories exist
    for directory in [DOCS_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(app, host=host, port=port)