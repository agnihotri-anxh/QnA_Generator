"""
Configuration settings for the QA Generator application.
These settings help manage memory usage and performance.
"""

import os

# Memory Management Settings
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", 5))
MAX_MEMORY_USAGE_MB = int(os.getenv("MAX_MEMORY_USAGE_MB", 300))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 1000))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 10))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 5))
MAX_PAGES = int(os.getenv("MAX_PAGES", 10))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 25000))

# API Settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 5.0))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 50))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 120))

# Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-saba-24b")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 2))

# File Settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/docs")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "static/output")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Performance Settings
ENABLE_EMBEDDINGS = os.getenv("ENABLE_EMBEDDINGS", "true").lower() == "true"
ENABLE_VECTOR_STORE = os.getenv("ENABLE_VECTOR_STORE", "true").lower() == "true"
GARBAGE_COLLECTION_INTERVAL = int(os.getenv("GARBAGE_COLLECTION_INTERVAL", 3)) 