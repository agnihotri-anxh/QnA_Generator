# Deployment Guide for QA Generator

## Option 1: Deploy to Render (Recommended)

### Step 1: Prepare Your Repository
1. Make sure your code is in a Git repository (GitHub, GitLab, etc.)
2. Ensure all files are committed and pushed

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name:** `qa-generator`
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python start_server.py`
   - **Plan:** Free

### Step 3: Set Environment Variables
In the Render dashboard, go to your service → Environment:
- Add `GROQ_API_KEY` with your actual Groq API key

### Step 4: Deploy
Click "Create Web Service" and wait for deployment.

---

## Option 2: Deploy to Railway

### Step 1: Prepare
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`

### Step 2: Deploy
```bash
railway init
railway up
```

### Step 3: Set Environment Variables
```bash
railway variables set GROQ_API_KEY=your_api_key_here
```

---

## Option 3: Deploy to Heroku

### Step 1: Install Heroku CLI
Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

### Step 2: Deploy
```bash
heroku create your-app-name
git push heroku main
```

### Step 3: Set Environment Variables
```bash
heroku config:set GROQ_API_KEY=your_api_key_here
```

---

## Option 4: Deploy to DigitalOcean App Platform

### Step 1: Prepare
1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click "Create App"

### Step 2: Configure
- Connect your Git repository
- Set build command: `pip install -r requirements.txt`
- Set run command: `python start_server.py`
- Add environment variable: `GROQ_API_KEY`

---

## Option 5: Deploy to Google Cloud Run

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "start_server.py"]
```

### Step 2: Deploy
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/qa-generator
gcloud run deploy qa-generator --image gcr.io/YOUR_PROJECT/qa-generator --platform managed
```

---

## Environment Variables Required

All deployments need these environment variables:

- `GROQ_API_KEY` - Your Groq API key (required)
- `PORT` - Server port (usually set automatically by platform)
- `HOST` - Server host (usually set automatically by platform)

---

## Post-Deployment

1. **Test your app** by visiting the provided URL
2. **Monitor logs** for any errors
3. **Set up custom domain** (optional)
4. **Configure SSL** (usually automatic)

---

## Troubleshooting

### Common Issues:

1. **Build fails:**
   - Check if all dependencies are in `requirements.txt`
   - Ensure Python version is compatible

2. **App crashes on startup:**
   - Check if `GROQ_API_KEY` is set correctly
   - Review application logs

3. **Memory issues:**
   - The app is optimized for memory, but you may need to upgrade your plan
   - Consider using smaller PDF files

4. **Port issues:**
   - Most platforms set `PORT` automatically
   - The app uses `os.getenv('PORT', 10000)` to handle this

---

## Recommended: Render Deployment

For the easiest deployment experience, I recommend **Render** because:
- Free tier available
- Automatic HTTPS
- Easy environment variable management
- Good Python support
- Automatic deployments from Git

Would you like me to help you with a specific platform deployment? 