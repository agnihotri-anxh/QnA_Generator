# QA Generator

A simple application that generates questions and answers from uploaded PDF documents using Groq's Mistral Saba 24B model.

## Features

- Upload PDF documents
- Generate questions and answers using Groq's Mistral Saba 24B model
- Download results as CSV
- Simple and clean web interface

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:10000`

## Usage

1. Upload a PDF document (max 5MB)
2. Click "Upload and Analyze"
3. Wait for processing to complete
4. View generated questions and answers
5. Download the CSV file

## Project Structure

```
QA_Generator/
├── app.py                 # Main FastAPI application
├── src/
│   ├── helper.py         # LLM pipeline and processing
│   └── prompt.py         # Prompt templates
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── docs/            # Uploaded documents
│   └── output/          # Generated CSV files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.8+
- Groq API key
- PDF documents to analyze

## Limitations

- Maximum file size: 5MB
- Maximum pages processed: 10
- Maximum questions generated: 5
- Supports PDF files only