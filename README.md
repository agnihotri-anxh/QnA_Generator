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
- **Memory optimization for cloud deployment**
- **Real-time memory monitoring**
- **Automatic resource cleanup**

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

## Memory Optimization

The application has been optimized for cloud deployment with memory constraints:

### Limits
- **Maximum document size**: 10MB
- **Maximum pages processed**: 20 pages
- **Maximum questions generated**: 10 questions
- **Memory usage limit**: 400MB (for Render's 512MB limit)

### Features
- Real-time memory monitoring
- Automatic garbage collection
- Fallback processing when memory is constrained
- Lighter ML models for better performance
- Configurable limits via environment variables

### Health Monitoring
Check application health and memory usage:
```bash
curl https://your-app.onrender.com/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1234567890,
  "memory_usage_mb": 245.67
}
```

## Deployment

### Render Deployment

1. **Environment Variables**:
   ```bash
   GROQ_API_KEY=your_groq_api_key
   MAX_DOCUMENT_SIZE_MB=10
   MAX_MEMORY_USAGE_MB=400
   ENABLE_EMBEDDINGS=true
   ```

2. **Build Configuration**:
   - The application automatically monitors memory usage
   - Rejects files larger than 10MB
   - Uses fallback mode if memory is high
   - Cleans up resources after processing

3. **Monitoring**:
   - Use the `/health` endpoint to monitor memory usage
   - Check logs for memory-related warnings
   - Monitor processing times

### Configuration Options

The application supports various configuration options via environment variables:

```bash
# Memory limits
MAX_DOCUMENT_SIZE_MB=10
MAX_MEMORY_USAGE_MB=400
MAX_CHUNKS=20
MAX_QUESTIONS=10
MAX_CHUNK_SIZE=2000

# Performance settings
ENABLE_EMBEDDINGS=true
ENABLE_VECTOR_STORE=true
TIMEOUT_SECONDS=180

# Model settings
EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
LLM_MODEL=mistral-saba-24b
```

For detailed memory optimization information, see [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md).

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
├── config.py              # Configuration settings
├── src/
│   ├── helper.py         # Helper functions and LLM pipeline
│   └── prompt.py         # Prompt templates
├── static/
│   ├── docs/            # Uploaded documents
│   └── output/          # Generated CSV files
├── templates/
│   └── index.html       # Web interface
├── requirements.txt      # Python dependencies
├── MEMORY_OPTIMIZATION.md # Memory optimization guide
└── .env                 # Environment variables
```

## Error Handling

The application handles various error scenarios:
- File upload errors
- API rate limits
- Token limit exceeded
- Invalid file formats
- Processing errors
- **Memory limit exceeded**
- **File size too large**
- **Processing timeouts**

## Troubleshooting

### Memory Issues
If you encounter memory issues:

1. **Reduce document size**: Use documents under 5MB
2. **Check memory usage**: Use the `/health` endpoint
3. **Disable embeddings**: Set `ENABLE_EMBEDDINGS=false`
4. **Reduce limits**: Lower `MAX_CHUNKS` and `MAX_QUESTIONS`

### Performance Issues
1. **Use smaller documents**: Process documents page by page
2. **Monitor logs**: Check for memory warnings
3. **Upgrade plan**: Consider upgrading your Render plan

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