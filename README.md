# Document Q&A Generator

A powerful application that generates questions and answers from uploaded documents using the Groq API. The application supports PDF, Word, and PowerPoint documents.

## Features

- Document upload support for multiple formats (PDF, DOCX, PPTX)
- Automatic question generation from document content
- Smart answer generation using Groq's LLM
- Interactive web interface with real-time feedback
- CSV export of generated Q&A pairs
- Rate limit handling and token usage tracking
- Document content preview
- Responsive design with modern UI

## Prerequisites

- Python 3.8 or higher
- Groq API key
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QA_Generator
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

3. Upload a document (PDF, Word, or PowerPoint)
4. Wait for the analysis to complete
5. View the generated questions and answers
6. Download the Q&A pairs as a CSV file

## Rate Limits and Token Usage

The application uses the Groq API, which has the following limits:
- Daily token limit: 500,000 tokens
- Rate limits apply to API calls

The application includes:
- Automatic token usage tracking
- Rate limit handling with exponential backoff
- User-friendly error messages
- Daily token usage reset
- Visual indicators for rate-limited responses

## Project Structure

```
QA_Generator/
├── app.py                 # Main application file
├── src/
│   ├── helper.py         # Helper functions and LLM pipeline
│   └── prompt.py         # Prompt templates
├── static/
│   ├── docs/            # Uploaded documents
│   └── output/          # Generated CSV files
├── templates/
│   └── index.html       # Web interface
├── requirements.txt      # Python dependencies
└── .env                 # Environment variables
```

## Error Handling

The application handles various error scenarios:
- File upload errors
- API rate limits
- Token limit exceeded
- Invalid file formats
- Processing errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Groq API for LLM capabilities
- LangChain for document processing
- Bootstrap for UI components
- PDF.js for PDF rendering