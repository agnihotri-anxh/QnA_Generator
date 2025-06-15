from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path
import shutil
import os
import logging
import uvicorn
from datetime import datetime
from src.helper import file_processing, llm_pipeline
from fastapi.middleware.cors import CORSMiddleware

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA Generator API")

# Add CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories with absolute paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = STATIC_DIR / "docs"
OUTPUT_DIR = STATIC_DIR / "output"

for directory in [DOCS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Mount static directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        allowed_types = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx'
        }
        
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF, Word, or PowerPoint file.")
        
        # Check file size (limit to 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
        file_size = 0
        chunk_size = 8192  # 8KB chunks
        
        # Read file in chunks to check size
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        
        # Reset file pointer
        await file.seek(0)
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = allowed_types[file.content_type]
        filename = f"{timestamp}_{file.filename}"
        file_path = DOCS_DIR / filename
        
        # Save the file in chunks
        try:
            with file_path.open("wb") as buffer:
                while chunk := await file.read(chunk_size):
                    buffer.write(chunk)
            logger.info(f"File uploaded successfully: {filename}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving file")
        finally:
            # Clean up
            await file.close()
        
        return JSONResponse(content={"success": True, "file_path": str(file_path)})
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_document(file_path: str = Form(...)):
    try:
        # Add detailed logging
        logger.info(f"Starting document analysis for path: {file_path}")
        logger.info(f"File exists: {os.path.exists(file_path)}")
        logger.info(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        logger.info(f"GROQ_API_KEY is set: {bool(os.getenv('GROQ_API_KEY'))}")
        
        # Validate file path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Process the document
        logger.info(f"Starting document analysis for: {file_path}")
        try:
            processed_chunks = file_processing(file_path)
            if not processed_chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
        # Generate Q&A
        logger.info("Starting Q&A generation")
        try:
            output_file, qa_list = llm_pipeline(processed_chunks, file_path)
        except Exception as e:
            logger.error(f"Error in LLM pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating Q&A: {str(e)}")
        
        # Clean up processed chunks
        del processed_chunks
        import gc
        gc.collect()
        
        logger.info(f"Document analyzed successfully: {file_path}")
        return JSONResponse(content={
            "success": True,
            "qa_list": qa_list,
            "csv_path": output_file,
            "document_content": processed_chunks,
            "file_name": os.path.basename(file_path)
        })
    
    except Exception as e:
        error_msg = f"Error analyzing document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    # Use localhost for local development, 0.0.0.0 for production
    host = os.getenv("HOST", "127.0.0.1")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True  # Enable reload for local development
    )