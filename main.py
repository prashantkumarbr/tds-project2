# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "httpx",
#   "pydantic",
#   "playwright",
#   "PyPDF2",
#   "pandas",
#   "openpyxl",
#   "xlrd",
#   "pillow",
#   "matplotlib",
#   "python-dotenv",
#   "python-multipart",
#   "numpy",
#   "scipy",
# ]
# ///

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
import httpx
import asyncio
from playwright.async_api import async_playwright, Browser
import base64
import json
import os
from typing import Optional, Any, Union
import re
from datetime import datetime
import logging
from io import BytesIO
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
YOUR_EMAIL = os.getenv("STUDENT_EMAIL", "your-email@example.com")
YOUR_SECRET = os.getenv("STUDENT_SECRET", "your-secret-string")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "your-aipipe-token")
MAX_PAYLOAD_SIZE = 1 * 1024 * 1024  # 1MB limit
QUIZ_TIMEOUT = 170  # 2:50 minutes, leaving 10s buffer

# Global browser instance for reuse
browser_instance: Optional[Browser] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage browser lifecycle"""
    global browser_instance
    playwright = await async_playwright().start()
    browser_instance = await playwright.chromium.launch(headless=True)
    logger.info("Browser started")
    yield
    await browser_instance.close()
    await playwright.stop()
    logger.info("Browser closed")

app = FastAPI(lifespan=lifespan)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ==================== ERROR HANDLERS ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 400 for invalid JSON or validation errors"""
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid JSON payload", "details": str(exc)}
    )

@app.exception_handler(json.JSONDecodeError)
async def json_exception_handler(request: Request, exc: json.JSONDecodeError):
    """Return 400 for malformed JSON"""
    return JSONResponse(
        status_code=400,
        content={"error": "Malformed JSON", "details": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ==================== LLM FUNCTIONS ====================

async def call_aipipe_llm(prompt: str, model: str = "openai/gpt-4.1-nano") -> str:
    """Call AIPipe API with improved response parsing"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://aipipe.org/openrouter/v1/responses",
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "input": prompt
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract text from various response formats
            if isinstance(data, dict):
                # New-style: data["output"][0]["content"][0]["text"]
                if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                    try:
                        first_output = data["output"][0]
                        content_list = first_output.get("content", [])
                        if isinstance(content_list, list) and len(content_list) > 0:
                            message = content_list[0]
                            if isinstance(message, dict):
                                extracted = message.get("text") or message.get("content") or ""
                                if isinstance(extracted, str) and extracted.strip():
                                    logger.info(f"✓ LLM response: {extracted[:150]}")
                                    return extracted.strip()
                    except Exception:
                        pass

                # Older: data["content"][0]["text"]
                if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                    first = data["content"][0]
                    if isinstance(first, dict) and "text" in first:
                        extracted = first.get("text", "").strip()
                        if extracted:
                            logger.info(f"✓ LLM response: {extracted[:150]}")
                            return extracted

                # OpenAI-style: choices
                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if isinstance(choice, dict) and "text" in choice:
                        extracted = choice.get("text", "").strip()
                        if extracted:
                            logger.info(f"✓ LLM response: {extracted[:150]}")
                            return extracted

                # Direct fields
                for key in ("text", "response", "result"):
                    if key in data and isinstance(data[key], str) and data[key].strip():
                        logger.info(f"✓ LLM response: {data[key][:150]}")
                        return data[key].strip()

            if isinstance(data, str) and data.strip():
                return data.strip()

            logger.error(f"Unexpected AIPipe response structure: {data}")
            return ""

    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return ""


# ==================== PAGE RENDERING ====================

async def fetch_and_render_page(url: str) -> tuple[str, str]:
    """Fetch and render a JavaScript page, return text content and HTML"""
    global browser_instance
    
    if not browser_instance:
        raise Exception("Browser not initialized")
    
    page = await browser_instance.new_page()
    
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)  # Wait for JS execution
        
        html_content = await page.content()
        text_content = await page.evaluate("document.body.innerText")
        
        return text_content, html_content
    finally:
        await page.close()

# ==================== FILE HANDLING ====================

async def download_file(url: str) -> tuple[bytes, str]:
    """Download a file and return bytes with content type"""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type

def file_to_base64(data: bytes, content_type: str) -> str:
    """Convert file bytes to base64 data URI"""
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:{content_type};base64,{b64}"

async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {page_num + 1} ---\n{page_text}\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# ==================== URL EXTRACTION ====================

def extract_urls_from_text(text: str, html: str, base_url: str = None) -> dict:
    """Extract all relevant URLs from page content"""
    urls = {
        "submit_url": None,
        "file_urls": [],
        "api_urls": []
    }
    
    # Extract submit URL - look for POST ... to <URL>
    submit_patterns = [
        r'POST.*?to\s+(https?://[^\s"\'<>]+)',
        r'submit.*?to\s+(https?://[^\s"\'<>]+)',
        r'POST.*?(https?://[^\s"\'<>]+/submit[^\s"\'<>]*)',
        r'POST.*?to.*?to\s+(https?://[^\s"\'<>]+)',  # Handle "POST to JSON to URL"
    ]
    
    for pattern in submit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            urls["submit_url"] = match.group(1).strip().rstrip('.,;:')
            break
    
    # Try relative paths
    if not urls["submit_url"] and base_url:
        relative_patterns = [
            r'POST.*?to\s+(/[^\s"\'<>]+)',
            r'submit.*?to\s+(/[^\s"\'<>]+)',
            r'POST.*?to.*?to\s+(/[^\s"\'<>]+)',  # Handle "POST to JSON to /submit"
        ]
        for pattern in relative_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                urls["submit_url"] = urljoin(base_url, match.group(1))
                break
    
    # Extract scrape URLs (API endpoints)
    scrape_patterns = [
        r'[Ss]crape\s+(/[^\s"\'<>]+)',
        r'[Ff]etch\s+(/[^\s"\'<>]+)',
        r'GET.*?from\s+(/[^\s"\'<>]+)'
    ]
    
    for pattern in scrape_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if base_url:
                full_url = urljoin(base_url, match)
                urls["api_urls"].append(full_url)
                logger.info(f"Found scrape URL: {full_url}")
    
    # Extract file URLs from HTML - more aggressive patterns
    file_patterns = [
        r'href=["\']?(https?://[^"\'>]+\.(?:csv|xlsx?|xls|pdf|json|txt))["\']?',
        r'src=["\']?(https?://[^"\'>]+\.(?:csv|xlsx?|xls|pdf|json|txt))["\']?',
        r'(https?://[^ \n\t"<>\']+\.(?:csv|xlsx?|xls|pdf|json|txt))',
        r'download[^<>]*href=["\']([^"\']+\.(?:csv|xlsx?|xls|pdf|json|txt))["\']',
        r'<a[^>]+href=["\']([^"\']+\.(?:csv|xlsx?|xls|pdf|json|txt))["\']',
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, html + " " + text, re.IGNORECASE)
        for match in matches:
            # Convert relative to absolute
            if not match.startswith('http'):
                if base_url:
                    match = urljoin(base_url, match)
            urls["file_urls"].append(match)
   
    # Also check for file mentions in text (e.g., "Download file.csv")
    text_file_pattern = r'\b([\w\-]+\.(?:csv|xlsx?|xls|pdf|json|txt))\b'
    text_files = re.findall(text_file_pattern, text, re.IGNORECASE)
    for filename in text_files:
        # Try to construct URL from base
        if base_url:
            potential_url = urljoin(base_url, filename)
            urls["file_urls"].append(potential_url)
            # Also try common paths
            for path in ['/', '/files/', '/data/', '/downloads/']:
                potential_url = urljoin(base_url, path + filename)
                urls["file_urls"].append(potential_url)
    
    # Relative file paths in HTML
    relative_file_matches = re.findall(
        r'href=["\'](/[^"\'>]+\.(?:csv|xlsx?|xls|pdf|json|txt))["\']',
        html,
        re.IGNORECASE
    )

    for rel in relative_file_matches:
        if base_url:
            full_url = urljoin(base_url, rel)
            urls["file_urls"].append(full_url)
            logger.info(f"Found file URL: {full_url}")

    # Remove duplicates
    urls["file_urls"] = list(set(urls["file_urls"]))
    urls["api_urls"] = list(set(urls["api_urls"]))
    
    return urls

# ==================== ANSWER EXTRACTION ====================

def extract_clean_answer(llm_response: str, expected_type: str = "string") -> Any:
    """Extract clean answer from LLM response, removing prefixes and formatting"""
    if not llm_response:
        return None
    
    # Remove common prefixes
    prefixes = [
        r'^ANSWER:\s*',
        r'^Answer:\s*',
        r'^The answer is:?\s*',
        r'^Result:?\s*',
        r'^Output:?\s*',
    ]
    
    cleaned = llm_response.strip()
    for prefix in prefixes:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = cleaned.strip().strip('"\'')
    
    # Try to parse based on expected type
    if expected_type == "number":
        # Extract first number found
        number_match = re.search(r'-?\d+\.?\d*', cleaned)
        if number_match:
            num_str = number_match.group(0)
            try:
                return int(num_str) if '.' not in num_str else float(num_str)
            except ValueError:
                pass
    
    elif expected_type == "boolean":
        if cleaned.lower() in ['true', 'yes', '1']:
            return True
        elif cleaned.lower() in ['false', 'no', '0']:
            return False
    
    elif expected_type == "json":
        # Try to find JSON in response
        json_match = re.search(r'[\{\[].*[\}\]]', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Return as string
    return cleaned

# ==================== TASK SOLVING ====================

async def solve_task_direct(task_text: str, urls: dict, base_url: str, html_content: str) -> Any:
    """Directly solve task without analysis step - faster and more reliable"""
    
    data_context = ""
    
    # 1. Process API URLs (scrape data from pages) - Use browser for JS rendering
    for api_url in urls.get("api_urls", []):
        try:
            logger.info(f"Scraping: {api_url}")
            
            # Use browser to render JavaScript
            scraped_text, scraped_html = await fetch_and_render_page(api_url)
            logger.info(f"📄 Scraped content: {scraped_text[:200]}")
            logger.info(f"🔍 Full scraped text for debugging: {scraped_text}")  # Log full content for debugging
            
            # Direct secret extraction - try multiple patterns
            secret_patterns = [
                r'secret[:\s]*([a-zA-Z0-9]{6,20})',
                r'code[:\s]*([a-zA-Z0-9]{6,20})',
                r'key[:\s]*([a-zA-Z0-9]{6,20})',
                r'\b([a-f0-9]{6,20})\b',  # Hex string
                r'\b([A-Z0-9]{6,20})\b',  # Alphanumeric uppercase
            ]
            
            for pattern in secret_patterns:
                secret_match = re.search(pattern, scraped_text, re.I)
                if secret_match:
                    secret = secret_match.group(1).strip()
                    # Avoid false positives
                    if secret.lower() not in ['secret', 'code', 'key', 'email', 'answer']:
                        logger.info(f"✓ Extracted secret: {secret}")
                        return secret
            
            data_context += f"\n\n=== Scraped from {api_url} ===\n{scraped_text}"
        except Exception as e:
            logger.error(f"Error scraping {api_url}: {e}")
    
    # 2. Download and process files
    if not urls.get("file_urls"):
        # Try to find file references in the task text itself
        logger.info("🔍 No file URLs found, checking task text for file references")
        
        # Look for patterns like "CSV file" or mentions of data files
        if re.search(r'\bCSV\b|\bcsv file\b|\bdata.*file\b', task_text, re.I):
            logger.info("📊 Task mentions CSV/data file - attempting to locate")
            
            # Try common file locations based on current URL
            parsed = urlparse(base_url)
            base_domain = f"{parsed.scheme}://{parsed.netloc}"
            
            # Extract any identifiers from the URL (like email, id)
            url_params = re.findall(r'[\?&](email|id)=([^&]+)', base_url)
            param_dict = {k: v for k, v in url_params}
            
            # Try common file naming patterns
            potential_files = []
            if 'email' in param_dict and 'id' in param_dict:
                # Pattern: /demo-audio-data.csv or /demo-audio-123.csv
                page_name = parsed.path.split('/')[-1].split('?')[0]
                potential_files.extend([
                    f"{base_domain}/{page_name}-data.csv",
                    f"{base_domain}/{page_name}.csv",
                    f"{base_domain}/data-{param_dict.get('id')}.csv",
                    f"{base_domain}/{page_name}-{param_dict.get('id')}.csv",
                ])
            
            # Try to download from potential locations
            for potential_url in potential_files:
                try:
                    logger.info(f"🔍 Trying: {potential_url}")
                    file_data, content_type = await download_file(potential_url)
                    logger.info(f"✅ Found CSV at: {potential_url}")
                    urls["file_urls"].append(potential_url)
                    break
                except Exception:
                    continue
    
    for file_url in urls.get("file_urls", []):
        try:
            logger.info(f"Downloading: {file_url}")
            file_data, content_type = await download_file(file_url)
            
            if "pdf" in content_type or file_url.lower().endswith(".pdf"):
                pdf_text = await extract_text_from_pdf(file_data)
                data_context += f"\n\n=== PDF: {file_url} ===\n{pdf_text}"
            
            elif "csv" in content_type or file_url.lower().endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(BytesIO(file_data))
                
                # Look for cutoff value in task
                cutoff_match = re.search(r'[Cc]utoff[:\s]*(\d+)', task_text)
                cutoff_value = int(cutoff_match.group(1)) if cutoff_match else None
                
                if cutoff_value:
                    logger.info(f"📊 Cutoff detected: {cutoff_value}")
                
                data_context += f"\n\n=== CSV: {file_url} ===\nShape: {df.shape}\nColumns: {list(df.columns)}\n"
                
                if cutoff_value:
                    data_context += f"Cutoff value: {cutoff_value}\n"
                
                data_context += f"First rows:\n{df.head(10).to_string()}\n\nLast rows:\n{df.tail(5).to_string()}\n\nSummary:\n{df.describe().to_string()}"
            
            elif "excel" in content_type or file_url.lower().endswith((".xlsx", ".xls")):
                import pandas as pd
                df = pd.read_excel(BytesIO(file_data))
                data_context += f"\n\n=== Excel: {file_url} ===\nShape: {df.shape}\nColumns: {list(df.columns)}\nFirst rows:\n{df.head(10).to_string()}\n\nSummary:\n{df.describe().to_string()}"
            
            elif "json" in content_type or file_url.lower().endswith(".json"):
                json_data = json.loads(file_data.decode('utf-8'))
                data_context += f"\n\n=== JSON: {file_url} ===\n{json.dumps(json_data, indent=2)[:3000]}"
            
        except Exception as e:
            logger.error(f"Error processing {file_url}: {e}")
    
    # 3. Build solving prompt
    solve_prompt = f"""You are a data analysis expert. Solve this task and provide ONLY the final answer value.

TASK:
{task_text}

DATA:
{data_context if data_context else "No external data provided - answer from task description"}

IMPORTANT INSTRUCTIONS:
- Read the task VERY carefully
- If there's a CSV with data and a cutoff value, you likely need to filter or sum based on that cutoff
- Common operations: sum, count, filter, aggregate
- Return ONLY the final numeric answer or text value
- NO explanations, NO "Answer:", NO extra words
- Just the value: e.g., 12345 or abc123

YOUR ANSWER:"""

    llm_response = await call_aipipe_llm(solve_prompt)
    
    if not llm_response:
        logger.error("Empty response from LLM")
        return "Unable to determine answer"
    
    # Determine expected type from task
    expected_type = "string"
    if re.search(r'\bsum\b|\btotal\b|\bcount\b|\bnumber\b', task_text, re.I):
        expected_type = "number"
    elif re.search(r'\btrue\b|\bfalse\b|\bboolean\b', task_text, re.I):
        expected_type = "boolean"
    
    answer = extract_clean_answer(llm_response, expected_type)
    logger.info(f"✓ Final answer: {answer} (type: {type(answer).__name__})")
    
    return answer

# ==================== QUIZ CHAIN HANDLER ====================

async def solve_quiz_chain(initial_url: str, email: str, secret: str):
    """Solve the complete quiz chain"""
    current_url = initial_url
    start_time = datetime.now()
    results = []
    
    while current_url:
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if elapsed > QUIZ_TIMEOUT:
            logger.warning("⏱️ Time limit reached")
            break
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing: {current_url}")
            logger.info(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            # Render the quiz page
            text_content, html_content = await fetch_and_render_page(current_url)
            logger.info(f"📄 Page content ({len(text_content)} chars):\n{text_content[:300]}...")
            
            # Extract URLs
            urls = extract_urls_from_text(text_content, html_content, current_url)
            logger.info(f"🔗 URLs found:")
            logger.info(f"  Submit: {urls['submit_url']}")
            logger.info(f"  Files: {urls['file_urls']}")
            logger.info(f"  APIs: {urls['api_urls']}")
            
            if not urls["submit_url"]:
                logger.error("❌ No submit URL found!")
                break
            
            # Solve the task directly
            answer = await solve_task_direct(text_content, urls, current_url, html_content)
            logger.info(f"💡 Answer: {answer}")
            
            # Build and validate payload
            payload = {
                "email": email,
                "secret": secret,
                "url": current_url,
                "answer": answer
            }
            
            payload_json = json.dumps(payload)
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"📦 Payload size: {payload_size} bytes")
            
            if payload_size > MAX_PAYLOAD_SIZE:
                logger.warning("⚠️ Payload too large, truncating...")
                if isinstance(answer, str) and len(answer) > 50000:
                    answer = answer[:50000]
                    payload["answer"] = answer
            
            # Submit the answer
            logger.info(f"📤 Submitting to: {urls['submit_url']}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    urls["submit_url"],
                    json=payload
                )
                
                try:
                    result = response.json()
                except Exception as e:
                    logger.error(f"❌ Invalid response: {e}")
                    result = {"correct": False, "reason": "Invalid JSON response"}
                
                results.append({
                    "url": current_url,
                    "answer": answer,
                    "result": result
                })
                
                if result.get("correct"):
                    logger.info(f"✅ Correct!")
                    next_url = result.get("url")
                    if next_url:
                        logger.info(f"➡️ Next quiz: {next_url}")
                        current_url = next_url
                    else:
                        logger.info("🎉 Quiz completed!")
                        break
                else:
                    reason = result.get("reason", "Unknown")
                    logger.warning(f"❌ Wrong: {reason}")
                    next_url = result.get("url")
                    if next_url:
                        logger.info(f"➡️ Moving to: {next_url}")
                        current_url = next_url
                    else:
                        logger.info("No next URL provided, stopping")
                        break
        
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            results.append({"url": current_url, "error": str(e)})
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Quiz Summary: {len(results)} questions attempted")
    return results

# ==================== API ENDPOINTS ====================

@app.post("/quiz")
async def handle_quiz(request: Request):
    """Main endpoint to receive quiz tasks"""
    
    try:
        body = await request.body()
        if not body:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty request body"}
            )
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "details": str(e)}
        )
    
    # Validate required fields
    if not all(k in data for k in ["email", "secret", "url"]):
        return JSONResponse(
            status_code=400,
            content={"error": "Missing required fields: email, secret, url"}
        )
    
    # Verify secret
    if data["secret"] != YOUR_SECRET:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid secret"}
        )
    
    # Verify email
    if data["email"] != YOUR_EMAIL:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid email"}
        )
    
    # Start solving asynchronously
    asyncio.create_task(
        solve_quiz_chain(data["url"], data["email"], data["secret"])
    )
    
    # Respond immediately with 200
    return JSONResponse(
        status_code=200,
        content={"status": "received", "message": "Quiz solving started"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "browser_ready": browser_instance is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LLM Quiz Solver API", "version": "2.0.0"}

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)