# QA Generator

A FastAPI-based application that generates questions and answers from PDF documents using Groq's LLM API.

## Features

- Upload PDF documents (max 5MB)
- Generate intelligent questions and answers
- Download results as CSV
- Modern, responsive UI
- Memory-optimized processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QA_Genrator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application

### Option 1: Using the optimized startup script (Recommended)
```bash
python start_server.py
```

### Option 2: Direct execution
```bash
python app.py
```

### Option 3: Using uvicorn directly
```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

The application will be available at `http://localhost:10000`

## Memory Optimization

The application includes several memory optimization features:

- Automatic garbage collection during processing
- Reduced chunk sizes and page limits
- Memory monitoring and warnings
- Single worker configuration
- Environment-specific optimizations

## Troubleshooting

### Common Issues

1. **405 Method Not Allowed for HEAD requests**
   - ✅ Fixed: Added HEAD method support

2. **404 Not Found for favicon.ico**
   - ✅ Fixed: Added favicon route

3. **Memory limit exceeded**
   - ✅ Fixed: Implemented memory optimization
   - Reduced processing limits
   - Added garbage collection

4. **LangChain deprecation warnings**
   - ✅ Fixed: Updated to use `invoke()` instead of `run()`

### Memory Issues

If you encounter memory issues:

1. Ensure you have at least 1GB of available RAM
2. Close other memory-intensive applications
3. Use the optimized startup script: `python start_server.py`
4. Consider reducing the file size of your PDF documents

### API Key Issues

If you get API key errors:

1. Ensure your `.env` file contains a valid `GROQ_API_KEY`
2. Get your API key from [Groq Console](https://console.groq.com/)
3. Restart the application after setting the API key

## File Structure

```
QA_Genrator/
├── app.py                 # Main FastAPI application
├── start_server.py        # Optimized startup script
├── requirements.txt       # Python dependencies
├── config.env            # Environment configuration
├── src/
│   ├── helper.py         # LLM processing logic
│   └── prompt.py         # Prompt templates
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── docs/             # Uploaded PDFs
│   └── output/           # Generated CSV files
└── uploads/              # Temporary upload directory
```

## API Endpoints

- `GET /` - Main application interface
- `POST /upload` - Upload PDF file
- `POST /analyze` - Generate Q&A from uploaded file
- `GET /download/{filename}` - Download generated CSV
- `GET /health` - Health check endpoint

## Environment Variables

- `GROQ_API_KEY` - Your Groq API key (required)
- `PORT` - Server port (default: 10000)
- `HOST` - Server host (default: 0.0.0.0)

## License

This project is licensed under the MIT License.