# Memory Optimization Guide

This document explains the memory optimizations implemented to prevent the "Ran out of memory" error on Render.

## Memory Issues Identified

The original application was experiencing memory issues due to:

1. **Large document processing**: Loading entire PDFs without size limits
2. **Heavy ML models**: Using large sentence transformer models
3. **Multiple LLM instances**: Creating separate instances for questions and answers
4. **Vector store creation**: FAISS vector store consuming significant memory
5. **No memory monitoring**: No constraints on document size or processing

## Optimizations Implemented

### 1. Memory Monitoring
- Added `psutil` for real-time memory usage tracking
- Implemented memory limits (400MB for Render's 512MB limit)
- Added garbage collection at strategic points

### 2. Document Size Limits
- Maximum document size: 10MB
- Maximum pages processed: 20 pages
- Maximum text length: 50,000 characters
- Maximum chunks: 20 chunks

### 3. Model Optimization
- Switched to lighter embedding model: `paraphrase-MiniLM-L3-v2`
- Reduced embedding batch size to 2
- Single LLM instance for both questions and answers
- Limited questions generated to 10

### 4. Processing Limits
- Reduced chunk size to 2,000 tokens
- Answer generation chunk size: 500 tokens
- Reduced timeout to 3 minutes
- Limited vector store documents to 10
- Added memory checks before processing

### 5. Garbage Collection
- Force garbage collection after file processing
- Garbage collection every 3 questions
- Memory cleanup after each major operation

## Configuration

The application uses environment variables for configuration:

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

## Deployment on Render

### 1. Environment Variables
Set these in your Render dashboard:

```bash
GROQ_API_KEY=your_groq_api_key
MAX_DOCUMENT_SIZE_MB=10
MAX_MEMORY_USAGE_MB=400
ENABLE_EMBEDDINGS=true
```

### 2. Build Configuration
The application will automatically:
- Monitor memory usage
- Reject files larger than 10MB
- Use fallback mode if memory is high
- Clean up resources after processing

### 3. Health Monitoring
The `/health` endpoint now includes memory usage:
```json
{
  "status": "healthy",
  "timestamp": 1234567890,
  "memory_usage_mb": 245.67
}
```

## Error Handling

The application now provides specific error messages:

- **File too large**: "File too large (15.2 MB). Maximum allowed: 10 MB"
- **Memory limit**: "Server is currently under high load. Please try again later."
- **Rate limit**: "API rate limit reached. Please try again later."
- **Timeout**: "Document processing timed out. Please try again."

## Monitoring

### Logs to Watch
- Memory usage before/after operations
- Garbage collection events
- File size validations
- Processing timeouts

### Key Metrics
- Memory usage should stay below 400MB
- Processing time should be under 3 minutes
- File uploads should be under 10MB

## Troubleshooting

### If Memory Issues Persist

1. **Reduce limits further**:
   ```bash
   MAX_DOCUMENT_SIZE_MB=5
   MAX_CHUNKS=10
   MAX_QUESTIONS=5
   ```

2. **Disable embeddings**:
   ```bash
   ENABLE_EMBEDDINGS=false
   ENABLE_VECTOR_STORE=false
   ```

3. **Use lighter models**:
   ```bash
   EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
   ```

### Performance Tuning

1. **For smaller documents**: Increase limits
2. **For better accuracy**: Enable embeddings (if memory allows)
3. **For faster processing**: Reduce question count

## Best Practices

1. **Test with small documents first**
2. **Monitor memory usage in logs**
3. **Use the health endpoint to check server status**
4. **Implement client-side file size validation**
5. **Consider upgrading Render plan for larger documents**

## Fallback Behavior

When memory is constrained, the application will:

1. Skip embedding generation
2. Use simple question generation
3. Process fewer chunks
4. Generate fewer questions
5. Force garbage collection more frequently

This ensures the application remains functional even under memory pressure. 