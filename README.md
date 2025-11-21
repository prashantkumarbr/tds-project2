LLM Analysis Quiz – Automated Quiz Solver API

This project is built for the IITM BS Data Science – LLM Analysis Quiz evaluation.
It automatically solves quiz tasks involving data sourcing, extraction, transformation, analysis, and visualization using:

FastAPI backend

Playwright (for JS-rendered pages)

Python data stack (Pandas, NumPy, PyPDF2, SciPy, Matplotlib, etc.)

AIPipe/OpenRouter LLM models

The system visits quiz URLs, extracts data from web pages / APIs / files, computes answers, and submits them within the 3-minute time limit.

🚀 Features
✔ Fully automated quiz solver

Handles multi-step quiz chains

Extracts text using Playwright

Scrapes JS-rendered pages

Downloads & processes CSV, Excel, JSON, PDF files

Supports numerical, boolean, text, JSON, and base64-encoded outputs

Submits answers automatically to the quiz endpoint

Keeps payload under 1MB

✔ Robust LLM integration

Calls AIPipe LLM API

Cleans LLM responses

Extracts numeric / boolean / JSON answers

✔ Smart data extraction

Scrapes URLs, relative paths, file names

Extracts "secret codes" from hidden pages

Handles malformed JSON

Detects cutoffs & performs required aggregations

Global browser instance for performance

✔ FastAPI backend

/quiz – main solving endpoint

/health – health check

/ – API info

Advanced error handling

400 for invalid JSON

403 for invalid email/secret

✔ Docker + Render deployment

Uses mcr.microsoft.com/playwright/python image

No need for manual browser installation

Works on Render Free tier

render.yaml + Dockerfile included

🧠 How It Works

The server receives a POST request:

{
  "email": "student email",
  "secret": "student secret",
  "url": "https://example.com/quiz-123"
}


It validates:

JSON format

Email matches environment variable

Secret matches environment variable

It launches a Playwright Chromium browser (headless)

It opens the quiz URL, extracts:

Page text

HTML

File URLs

API URLs

Submit URL

It downloads and processes any required files.

It formats a solving prompt and sends it to an LLM.

It extracts a clean answer (number/string/bool/json).

It submits the answer to the page’s submit URL.

If the server returns a new URL, it repeats the process.

Completes the entire quiz chain within 3 minutes.

📡 API Endpoints
POST /quiz

Start solving a quiz chain.

Payload:
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://example.com/quiz-123"
}

GET /health

Health check:

{
  "status": "healthy",
  "timestamp": "...",
  "browser_ready": true
}

GET /

Root metadata:

{
  "message": "LLM Quiz Solver API",
  "version": "2.0.0"
}

🧩 Environment Variables

Create a .env file:

STUDENT_EMAIL=your-email@example.com
STUDENT_SECRET=your-secret-code
AIPIPE_TOKEN=your-aipipe-openrouter-token
PORT=10000

🐳 Docker Support

This project uses the official Playwright Python image.

Dockerfile:

FROM mcr.microsoft.com/playwright/python:latest
WORKDIR /app

COPY requirements.txt .
RUN sed -i '/playwright/d' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

☁️ Deployment on Render

This repository includes:

render.yaml

Dockerfile

Push to GitHub → Render auto-deploys.

Your URL stays fixed even after redeployments.

🧪 Testing

Send test query:

curl -X POST https://your-render-url/quiz \
  -H "Content-Type: application/json" \
  -d '{
        "email": "your email",
        "secret": "your secret",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
      }'

📝 Project Structure
├── main.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── README.md
└── .env (not committed)

📖 Viva Preparation (Important)

The LLM evaluator may ask:

System architecture

Why Playwright (JS execution)

Why async FastAPI

Why reuse global browser instance

Data handling

How you extract URLs

How you process CSV, Excel, PDF, JSON

How you ensure under 1MB payload

LLM strategy

How you clean LLM responses

How you avoid hallucinations

Why you avoid explanations

Error handling

400 → invalid JSON

403 → invalid secret/email

500 → unexpected exceptions

Prepare answers based on your code logic.

📄 License

This project is licensed under the MIT License.
See the LICENSE file.

👤 Author

Prashant Kumar
Full-stack & automation enthusiast
Repo: https://github.com/prashantkumarbr
