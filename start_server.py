#!/usr/bin/env python3
"""
Startup script for QA Generator with memory optimization
"""
import os
import sys
import gc
import psutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_memory():
    """Check available memory and log it"""
    try:
        memory = psutil.virtual_memory()
        logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
        logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
        logger.info(f"Memory usage: {memory.percent:.1f}%")
        
        if memory.available < 300 * 1024 * 1024:  # Less than 300MB
            logger.warning("Very low memory available! This may cause issues.")
            return False
        elif memory.available < 500 * 1024 * 1024:  # Less than 500MB
            logger.warning("Low memory available! Consider closing other applications.")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check memory: {e}")
        return True

def optimize_memory():
    """Perform memory optimization"""
    logger.info("Performing memory optimization...")
    
    # Force garbage collection multiple times
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Set memory limits for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['HF_HOME'] = '/tmp/huggingface'
    
    # Check if we're on a memory-constrained environment
    if os.getenv('RENDER', False) or os.getenv('HEROKU', False):
        logger.info("Detected cloud environment - applying aggressive memory optimizations")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Set lower memory limits for cloud environments
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_DATASETS_CACHE'] = '/tmp/datasets'
        
    # Additional optimizations
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    logger.info("Memory optimization completed")

def main():
    """Main startup function"""
    try:
        logger.info("Starting QA Generator application...")
        
        # Check memory before starting
        if not check_memory():
            logger.warning("Continuing with low memory warning")
        
        # Optimize memory
        optimize_memory()
        
        # Import and run the app
        from app import app
        import uvicorn
        
        # Get port from environment or use default
        port = int(os.getenv('PORT', 10000))
        host = os.getenv('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        
        # Run with optimized settings for memory-constrained environments
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            workers=1,  # Single worker to reduce memory usage
            loop="asyncio",
            limit_concurrency=10,  # Limit concurrent connections
            limit_max_requests=100,  # Restart worker after 100 requests
            timeout_keep_alive=30,  # Reduce keep-alive timeout
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 