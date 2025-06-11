# Question-Answer Generator

A powerful application that generates questions and answers from various document types (PDF, Word, PowerPoint) using advanced language models.

## Features

- Support for multiple document types (PDF, Word, PowerPoint)
- Interactive PDF viewer with zoom and navigation controls
- Automatic question generation from document content
- Detailed answers with source references
- Export results to CSV format
- Modern and responsive user interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/agnihotri-anxh/QA_Generator.git
cd QA_Generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Upload a document (PDF, Word, or PowerPoint)
2. Wait for the document to be processed
3. View the generated questions and answers
4. Download the complete Q&A in CSV format

## Requirements

- Python 3.8+
- FastAPI
- LangChain
- PDF.js (included)
- Other dependencies listed in requirements.txt

## License

MIT License