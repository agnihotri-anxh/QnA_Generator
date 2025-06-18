#!/bin/bash

# QA Generator Deployment Script
echo "🚀 QA Generator Deployment Script"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  Warning: GROQ_API_KEY environment variable is not set"
    echo "   You'll need to set it in your deployment platform"
fi

echo ""
echo "📋 Available deployment options:"
echo "1. Render (Recommended - Free tier)"
echo "2. Railway"
echo "3. Heroku"
echo "4. DigitalOcean App Platform"
echo "5. Google Cloud Run"
echo "6. Docker"
echo ""

read -p "Choose deployment option (1-6): " choice

case $choice in
    1)
        echo "🎯 Deploying to Render..."
        echo "1. Go to https://render.com"
        echo "2. Sign up/login and click 'New +' → 'Web Service'"
        echo "3. Connect your Git repository"
        echo "4. Configure:"
        echo "   - Name: qa-generator"
        echo "   - Environment: Python"
        echo "   - Build Command: pip install -r requirements.txt"
        echo "   - Start Command: python start_server.py"
        echo "5. Add environment variable: GROQ_API_KEY"
        echo "6. Click 'Create Web Service'"
        ;;
    2)
        echo "🚂 Deploying to Railway..."
        echo "Installing Railway CLI..."
        npm install -g @railway/cli
        echo "Login to Railway..."
        railway login
        echo "Deploying..."
        railway init
        railway up
        echo "Set environment variable:"
        echo "railway variables set GROQ_API_KEY=your_api_key_here"
        ;;
    3)
        echo "🦸 Deploying to Heroku..."
        echo "Make sure you have Heroku CLI installed"
        echo "heroku create your-app-name"
        echo "git push heroku main"
        echo "heroku config:set GROQ_API_KEY=your_api_key_here"
        ;;
    4)
        echo "🐙 Deploying to DigitalOcean App Platform..."
        echo "1. Go to https://cloud.digitalocean.com/apps"
        echo "2. Click 'Create App'"
        echo "3. Connect your Git repository"
        echo "4. Set build command: pip install -r requirements.txt"
        echo "5. Set run command: python start_server.py"
        echo "6. Add environment variable: GROQ_API_KEY"
        ;;
    5)
        echo "☁️  Deploying to Google Cloud Run..."
        echo "Make sure you have gcloud CLI installed and configured"
        echo "gcloud builds submit --tag gcr.io/YOUR_PROJECT/qa-generator"
        echo "gcloud run deploy qa-generator --image gcr.io/YOUR_PROJECT/qa-generator --platform managed"
        ;;
    6)
        echo "🐳 Deploying with Docker..."
        echo "Building Docker image..."
        docker build -t qa-generator .
        echo "Running container..."
        docker run -p 8080:8080 -e GROQ_API_KEY=your_api_key_here qa-generator
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-6."
        exit 1
        ;;
esac

echo ""
echo "✅ Deployment instructions completed!"
echo "📖 For detailed instructions, see DEPLOYMENT.md"
echo "🔧 For troubleshooting, check the logs in your deployment platform" 