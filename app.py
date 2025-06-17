from fastapi import FastAPI, Form, Request, Response, File, HTTPException
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
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = STATIC_DIR / "docs"
OUTPUT_DIR = STATIC_DIR / "output"

for directory in [DOCS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(pdf_file: bytes = File(), filename: str = Form(...)):
    try:
        # Check file size (5MB limit)
        file_size_mb = len(pdf_file) / (1024 * 1024)
        if file_size_mb > 5:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.2f} MB). Maximum allowed: 5 MB"
            )
        
        # Validate file extension
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        pdf_filename = os.path.join("static/docs", filename)
        
        async with aiofiles.open(pdf_filename, 'wb') as f:
            await f.write(pdf_file)
        
        logger.info(f"File uploaded successfully: {filename}")
        response_data = jsonable_encoder(json.dumps({"msg": 'success', "pdf_filename": pdf_filename}))
        return Response(response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def get_csv(file_path):
    try:
        logger.info(f"Starting analysis for file: {file_path}")
        answer_generation_chain, ques_list = llm_pipeline(file_path)
        
        if not ques_list:
            raise Exception("No questions were generated from the document")
        
        output_filename = f"{Path(file_path).stem}_QA.csv"
        output_file = OUTPUT_DIR / output_filename
        
        # Generate all questions and answers for CSV
        all_qa_data = []
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Question", "Answer"])
            
            for i, question in enumerate(ques_list):
                logger.info(f"Processing question {i+1}/{len(ques_list)}: {question[:50]}...")
                try:
                    answer = answer_generation_chain.run(question)
                    if not answer or answer.strip() == "":
                        answer = "No answer generated for this question."
                    
                    all_qa_data.append({"question": question, "answer": answer})
                    csv_writer.writerow([question, answer])
                    logger.info(f"Question {i+1} processed successfully")
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {str(e)}")
                    all_qa_data.append({"question": question, "answer": f"Error generating answer: {str(e)}"})
                    csv_writer.writerow([question, f"Error generating answer: {str(e)}"])
        
        # Return only first 5 questions for UI display
        display_qa_data = all_qa_data[:5]
        
        logger.info(f"Analysis completed. Generated {len(all_qa_data)} Q&A pairs for CSV, displaying {len(display_qa_data)} in UI")
        return str(output_file), display_qa_data
    except Exception as e:
        logger.error(f"Error in get_csv: {str(e)}")
        raise Exception(f"Error generating Q&A: {str(e)}")

@app.post("/analyze")
async def analyze_document(pdf_filename: str = Form(...)):
    try:
        if not os.path.exists(pdf_filename):
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        
        logger.info(f"Starting document analysis: {pdf_filename}")
        output_file, qa_data = get_csv(pdf_filename)
        
        response_data = jsonable_encoder(json.dumps({
            "output_file": output_file,
            "qa_data": qa_data,
            "status": "success"
        }))
        return Response(response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Download file not found")
        return FileResponse(file_path, filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    host = "localhost"
    port = 10000
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)