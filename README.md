# QA Generator

A FastAPI-based application that generates questions and answers from PDF documents using Groq's LLM API.

## Features

- Upload PDF documents (max 5MB)
- Generate intelligent questions and answers
- Download results as CSV
- Modern, responsive UI
- Memory-optimized processing
- Deployed on Vercel with global CDN

## Live Demo

Your app is deployed on Vercel! 🚀

## Installation (Local Development)

1. Clone the repository:
```bash
git clone https://github.com/agnihotri-anxh/QnA_Generator.git
cd QnA_Generator
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

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:10000`

## Deploy to Vercel

### Quick Deployment Steps:

1. **Go to [vercel.com](https://vercel.com)**
2. **Sign up/Login with GitHub**
3. **Click "New Project"**
4. **Import your GitHub repository** (`agnihotri-anxh/QnA_Generator`)
5. **Configure project:**
   - Framework Preset: `Other`
   - Root Directory: `./`
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `./`
6. **Click "Deploy"**
7. **Add Environment Variables:**
   - Go to Project Settings → Environment Variables
   - Add `GROQ_API_KEY` with your actual Groq API key
8. **Redeploy**

### Vercel Configuration

The project includes `vercel.json` for optimal Vercel deployment:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ],
  "env": {
    "PYTHONPATH": "."
  }
}
```

## Memory Optimization

The application includes several memory optimization features:

- Automatic garbage collection during processing
- Reduced chunk sizes and page limits
- Memory monitoring and warnings
- Environment-specific optimizations
- Optimized for Vercel's serverless environment

## File Structure

```
QA_Genrator/
├── app.py                 # Main FastAPI application
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── helper.py         # LLM processing logic
│   └── prompt.py         # Prompt templates
├── templates/
│   └── index.html        # Web interface
└── static/               # Static files
    ├── docs/             # Uploaded PDFs
    └── output/           # Generated CSV files
```

## API Endpoints

- `GET /` - Main application interface
- `POST /upload` - Upload PDF file
- `POST /analyze` - Generate Q&A from uploaded file
- `GET /download/{filename}` - Download generated CSV
- `GET /health` - Health check endpoint

## Environment Variables

- `GROQ_API_KEY` - Your Groq API key (required)
- `PORT` - Server port (set automatically by Vercel)
- `HOST` - Server host (set automatically by Vercel)

## Vercel Advantages

- ✅ **Global CDN** - Fast worldwide access
- ✅ **Automatic HTTPS** - Secure by default
- ✅ **Serverless** - Pay only for usage
- ✅ **Easy deployment** - Git integration
- ✅ **Good Python support** - FastAPI compatible
- ✅ **Free tier** - Generous limits

## Troubleshooting

### Common Issues:

1. **Build fails:**
   - Check if all dependencies are in `requirements.txt`
   - Ensure Python version is compatible

2. **Environment variables not working:**
   - Redeploy after adding environment variables
   - Check variable names are correct

3. **Memory issues:**
   - Use smaller PDF files (1-3 pages)
   - The app is optimized for Vercel's memory limits

4. **Timeout errors:**
   - Use smaller documents
   - The app is optimized for quick processing

## License

This project is licensed under the MIT License.