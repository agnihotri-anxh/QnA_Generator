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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
Path("static/docs").mkdir(parents=True, exist_ok=True)
Path("static/output").mkdir(parents=True, exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = allowed_types[file.content_type]
        filename = f"{timestamp}_{file.filename}"
        file_path = Path("static/docs") / filename
        
        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded successfully: {filename}")
        return JSONResponse(content={"success": True, "file_path": str(file_path)})
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_document(file_path: str = Form(...)):
    try:
        # Process the document
        logger.info(f"Starting document analysis for: {file_path}")
        processed_chunks = file_processing(file_path)
        if not processed_chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
        # Generate Q&A
        logger.info("Starting Q&A generation")
        output_file, qa_list = llm_pipeline(processed_chunks, file_path)
        
        logger.info(f"Document analyzed successfully: {file_path}")
        return JSONResponse(content={
            "success": True,
            "qa_list": qa_list,
            "csv_path": output_file,
            "document_content": processed_chunks
        })
    
    except Exception as e:
        error_msg = f"Error analyzing document: {str(e)}"
        logger.error(error_msg, exc_info=True)  # This will log the full stack trace
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = Path("static/output") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )