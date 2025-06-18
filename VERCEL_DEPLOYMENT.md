# Deploy QA Generator to Vercel

## 🚀 Quick Deployment Steps

### Method 1: Using Vercel CLI (Recommended)

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy from your project directory**
   ```bash
   vercel
   ```

4. **Follow the prompts:**
   - Set up and deploy? → `Y`
   - Which scope? → Select your account
   - Link to existing project? → `N`
   - Project name? → `qa-generator` (or your preferred name)
   - Directory? → `./` (current directory)
   - Override settings? → `N`

5. **Set Environment Variables**
   ```bash
   vercel env add GROQ_API_KEY
   # Enter your Groq API key when prompted
   ```

6. **Redeploy with environment variables**
   ```bash
   vercel --prod
   ```

### Method 2: Using Vercel Dashboard

1. **Go to [vercel.com](https://vercel.com)**
2. **Sign up/Login with GitHub**
3. **Click "New Project"**
4. **Import your GitHub repository**
5. **Configure project:**
   - Framework Preset: `Other`
   - Root Directory: `./`
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `./`
   - Install Command: `pip install -r requirements.txt`
6. **Click "Deploy"**
7. **Add Environment Variables:**
   - Go to Project Settings → Environment Variables
   - Add `GROQ_API_KEY` with your actual API key
8. **Redeploy**

## ⚙️ Configuration Files

### `vercel.json`
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
  },
  "functions": {
    "app.py": {
      "maxDuration": 300
    }
  }
}
```

## 🔧 Environment Variables

Set these in your Vercel project:

| Variable | Value | Required |
|----------|-------|----------|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes |
| `PYTHONPATH` | `.` | ✅ Yes |

## 📁 Project Structure for Vercel

```
QA_Genrator/
├── app.py                 # Main FastAPI app
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── api/
│   └── index.py         # Vercel serverless entry point
├── src/
│   ├── helper.py        # LLM processing logic
│   └── prompt.py        # Prompt templates
├── templates/
│   └── index.html       # Web interface
└── static/              # Static files
```

## 🎯 Vercel Advantages

- ✅ **Global CDN** - Fast worldwide access
- ✅ **Automatic HTTPS** - Secure by default
- ✅ **Serverless** - Pay only for usage
- ✅ **Easy deployment** - Git integration
- ✅ **Good Python support** - FastAPI compatible
- ✅ **Free tier** - Generous limits

## ⚠️ Important Notes for Vercel

### **Memory Limitations**
- Vercel functions have memory limits
- Your app is optimized for low memory usage
- Consider using smaller PDFs for testing

### **Function Timeout**
- Default timeout: 10 seconds
- Extended to 300 seconds in `vercel.json`
- For long-running tasks, consider background jobs

### **File Upload Limits**
- Vercel has file size limits
- Your app limits PDFs to 5MB
- This should work fine with Vercel

## 🔍 Troubleshooting

### **Common Issues:**

1. **Build fails:**
   - Check `requirements.txt` is up to date
   - Ensure all dependencies are compatible

2. **Environment variables not working:**
   - Redeploy after adding environment variables
   - Check variable names are correct

3. **Memory issues:**
   - Use smaller PDF files
   - Check Vercel function logs

4. **Timeout errors:**
   - Increase `maxDuration` in `vercel.json`
   - Optimize processing time

### **Check Logs:**
```bash
vercel logs
```

## 🚀 After Deployment

1. **Test your app** at the provided Vercel URL
2. **Monitor performance** in Vercel dashboard
3. **Set up custom domain** (optional)
4. **Configure analytics** (optional)

## 📞 Support

- **Vercel Documentation:** [vercel.com/docs](https://vercel.com/docs)
- **Vercel Community:** [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)

---

**Your app is ready for Vercel deployment!** 🎉 