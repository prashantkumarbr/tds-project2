# LLM Analysis Quiz – Automated Quiz Solver API

This project is built for the **IITM BS Data Science – LLM Analysis Quiz** evaluation.  
It automatically solves quiz tasks involving data sourcing, extraction, analysis, and visualization using:

- FastAPI
- Playwright (for JavaScript-rendered pages)
- AIPipe/OpenRouter LLM models
- Python data stack: Pandas, NumPy, PyPDF2, SciPy, Matplotlib, etc.

The system visits quiz URLs, extracts data, computes answers using LLM + Python, and submits results within the required 3-minute window.

---

## 🚀 Features

### ✔ Fully automated quiz solver
- Handles **multi-step** quiz chains
- Extracts text & HTML using Playwright
- Scrapes JS-rendered websites
- Downloads & processes CSV, Excel, JSON, PDF files
- Supports numeric, boolean, string, JSON, and file outputs
- Payload always under **1MB**

### ✔ LLM Integration
- Calls AIPipe models (e.g., GPT-4.1-nano)
- Cleans responses & extracts final answer
- Auto-detects numeric/boolean/string types

### ✔ Smart data extraction
- Detects submit URLs, file URLs, API URLs
- Converts relative → absolute links
- Extracts secrets from hidden pages
- Performs cutoff-based filtering & aggregates

### ✔ FastAPI Backend
- `/quiz` – starts solving
- `/health` – health status
- `/` – API metadata
- 400 → invalid JSON  
- 403 → invalid email/secret  
- Global exception handling

### ✔ Docker & Render Deployment
- Uses Playwright official image  
- No manual browser install required  
- Works on Render Free tier  
- Includes `Dockerfile` + `render.yaml`

---

## 📡 API Endpoints

### **POST /quiz**

```json
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://example.com/quiz-123"
}
GET /health
json
Copy code
{
  "status": "healthy",
  "timestamp": "...",
  "browser_ready": true
}
GET /
json
Copy code
{
  "message": "LLM Quiz Solver API",
  "version": "2.0.0"
}
🧪 Test Your Endpoint
Use this payload to test locally or on Render:

json
Copy code
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
Command:

bash
Copy code
curl -X POST https://your-render-url/quiz \
  -H "Content-Type: application/json" \
  -d '{"email":"your email","secret":"your secret","url":"https://tds-llm-analysis.s-anand.net/demo"}'
🧠 How It Works
Receive POST request

Verify:

JSON validity

Email matches

Secret matches

Launch global Playwright browser

Visit quiz URL

Extract:

Page text

HTML

Submit URL

File URLs

API URLs

Download & process files

Call LLM for final answer

Clean LLM output

Submit answer

If new URL provided → repeat until quiz ends

All within < 3 minutes.

🔧 Environment Variables
Create .env:

ini
Copy code
STUDENT_EMAIL=your-email@example.com
STUDENT_SECRET=your-secret-code
AIPIPE_TOKEN=your-aipipe-token
PORT=10000
🐳 Docker Deployment
The project uses the Playwright Python base image.

Example Dockerfile:

dockerfile
Copy code
FROM mcr.microsoft.com/playwright/python:latest
WORKDIR /app

COPY requirements.txt .
RUN sed -i '/playwright/d' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
☁️ Render Deployment
The repository includes:

render.yaml

Dockerfile

Push to GitHub → Render auto-deploys.
Your Render URL does not change on redeploy.

📁 Project Structure
bash
Copy code
├── main.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── README.md
└── .env (ignored)
🎤 Viva Preparation
You may be asked:

Architecture
Why Playwright?

Why async FastAPI?

Why use a global browser instance?

Data Handling
How URLs & files are extracted

How CSV/PDF/Excel/JSON processing works

How payload stays <1MB

LLM Strategy
How answer extraction works

How hallucinations are prevented

Error Handling
400 for invalid JSON

403 for secret/email mismatch

500 global errors

📄 License
Licensed under the MIT License.
See LICENSE.

👤 Author
Prashant Kumar
GitHub: https://github.com/prashantkumarbr
