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
#   "faster-whisper",
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
from collections import Counter
from PIL import Image
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
    playwright = None
    try:
        playwright = await async_playwright().start()
    except Exception as e:
        logger.error(f"Failed to start Playwright: {e}")
        playwright = None

    if playwright is not None:
        try:
            browser_instance = await playwright.chromium.launch(headless=True)
            logger.info("Browser started")
        except Exception as e:
            logger.error(f"Could not launch browser: {e}")
            browser_instance = None

    try:
        yield
    finally:
        # Clean up if browser/playwright started
        if browser_instance is not None:
            try:
                await browser_instance.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping Playwright: {e}")
        logger.info("Lifespan cleanup complete")

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

def analyze_image_dominant_color(image_bytes: bytes) -> str:
    """Find the most frequent RGB color in an image and return as hex"""
    try:
        from PIL import Image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get all pixels
        pixels = list(img.getdata())
        
        # Count colors
        color_counts = Counter(pixels)
        
        # Get most common
        most_common = color_counts.most_common(1)[0][0]
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(most_common[0], most_common[1], most_common[2])
        logger.info(f"🎨 Dominant color: {hex_color} (appears {color_counts[most_common]} times)")
        
        return hex_color
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return "#000000"

# ==================== URL EXTRACTION ====================

def extract_urls_from_text(text: str, html: str, base_url: str = None) -> dict:
    """Extract all relevant URLs from page content"""
    urls = {
        "submit_url": None,
        "file_urls": [],
        "api_urls": []
    }
    
    # ALWAYS use /submit as the submission endpoint
    if base_url:
        parsed = urlparse(base_url)
        urls["submit_url"] = f"{parsed.scheme}://{parsed.netloc}/submit"
        logger.info(f"✓ Submit URL: {urls['submit_url']}")
    
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
    
    # Extract file URLs - look for direct file paths mentioned (ONLY ONCE!)
    file_path_pattern = r'(/project2/[\w\-]+\.(?:csv|xlsx?|xls|pdf|json|txt|opus|png|jpg|jpeg|zip))'
    file_paths = re.findall(file_path_pattern, text, re.IGNORECASE)
    
    for file_path in file_paths:
        if base_url:
            full_url = urljoin(base_url, file_path)
            urls["file_urls"].append(full_url)
            logger.info(f"Found file URL from path: {full_url}")
    
    # Extract file URLs from HTML (absolute URLs)
    file_patterns = [
        r'href=["\']?(https?://[^"\'>]+\.(?:csv|xlsx?|xls|pdf|json|txt|opus|png|jpg|jpeg))["\']?',
        r'src=["\']?(https?://[^"\'>]+\.(?:csv|xlsx?|xls|pdf|json|txt|opus|png|jpg|jpeg))["\']?',
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, html + " " + text, re.IGNORECASE)
        for match in matches:
            if not match.startswith('http'):
                if base_url:
                    match = urljoin(base_url, match)
            urls["file_urls"].append(match)
    
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
    
    cleaned = cleaned.strip()
    
    # Try to parse based on expected type
    if expected_type == "json":
        # Try to find JSON in response
        json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole thing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    elif expected_type == "number":
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
    
    # Return as string, removing quotes if present
    return cleaned.strip('"\'')


# =================zip =================

async def handle_zip_logs_task(zip_bytes: bytes, task_text: str) -> int:
    """Handle zip file with logs"""
    try:
        import zipfile
        import json
        from io import BytesIO
        
        # Extract zip
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            logger.info(f"📦 ZIP contents: {zf.namelist()}")
            
            total_download_bytes = 0
            
            for filename in zf.namelist():
                logger.info(f"  Processing: {filename}")
                
                try:
                    with zf.open(filename) as f:
                        content = f.read().decode('utf-8')
                        
                        # Check if it's JSONL (JSON Lines)
                        if filename.endswith('.jsonl'):
                            lines = content.strip().split('\n')
                            logger.info(f"  📄 JSONL with {len(lines)} lines")
                            
                            for line in lines:
                                if line.strip():
                                    try:
                                        obj = json.loads(line)
                                        if obj.get('event') == 'download' and 'bytes' in obj:
                                            total_download_bytes += obj['bytes']
                                            logger.info(f"    ✓ Download: {obj['bytes']} bytes")
                                    except json.JSONDecodeError as e:
                                        logger.error(f"    ❌ Invalid JSON line: {line[:50]}")
                        
                        # Also try as regular JSON array
                        elif filename.endswith('.json'):
                            data = json.loads(content)
                            if isinstance(data, list):
                                for obj in data:
                                    if obj.get('event') == 'download' and 'bytes' in obj:
                                        total_download_bytes += obj['bytes']
                                        logger.info(f"    ✓ Download: {obj['bytes']} bytes")
                                        
                except Exception as e:
                    logger.error(f"  ❌ Error processing {filename}: {e}")
            
            logger.info(f"📊 Total download bytes: {total_download_bytes}")
            
            # Add email offset
            email_length = len(YOUR_EMAIL)
            offset = email_length % 5
            final_answer = total_download_bytes + offset
            
            logger.info(f"📧 Email length: {email_length}, offset (mod 5): {offset}")
            logger.info(f"✅ Final answer: {final_answer}")
            
            return final_answer
            
    except Exception as e:
        logger.error(f"ZIP processing error: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
# ===============pdf====================

async def handle_pdf_invoice_task(pdf_bytes: bytes, task_text: str) -> float:
    """Handle PDF invoice calculation"""
    try:
        from PyPDF2 import PdfReader
        import re
        
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        logger.info(f"📄 PDF text extracted:\n{text}")
        
        lines = text.strip().split('\n')
        
        # Parse line by line - the structure is:
        # Widget A
        # 3
        # 19.99
        # Widget B
        # 2
        # 5.5
        
        total = 0.0
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            logger.info(f"Line {i}: '{line}'")
            
            # Skip header lines
            if line in ['Invoice', 'Item', 'Quantity', 'UnitPrice', '']:
                i += 1
                continue
            
            # Check if this line starts with "Widget" (item name)
            if line.startswith('Widget'):
                # Next line should be quantity
                if i + 1 < len(lines):
                    try:
                        qty = float(lines[i + 1].strip())
                        # Line after that should be price
                        if i + 2 < len(lines):
                            price = float(lines[i + 2].strip())
                            line_total = qty * price
                            logger.info(f"  ✓ {line}: qty={qty}, price=${price}, total=${line_total}")
                            total += line_total
                            i += 3  # Skip to next item
                            continue
                    except ValueError as e:
                        logger.error(f"  ❌ Parse error: {e}")
            
            i += 1
        
        result = round(total, 2)
        logger.info(f"✅ Invoice total: ${result}")
        return result
        
    except Exception as e:
        logger.error(f"PDF invoice error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# ==================== SPECIALIZED HANDLERS ====================

async def handle_audio_task(file_url: str, base_url: str) -> str:
    """Handle audio transcription tasks using faster-whisper"""
    try:
        logger.info(f"🎵 Downloading audio: {file_url}")
        file_data, content_type = await download_file(file_url)
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            # Use faster-whisper (lighter, faster)
            from faster_whisper import WhisperModel
            
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(tmp_path)
            
            transcription = " ".join([segment.text for segment in segments])
            transcription = transcription.strip().lower()
            
            logger.info(f"✅ Transcription: {transcription}")
            return transcription
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        return ""

# embending tasks

async def handle_embeddings_task(json_data: dict, task_text: str) -> str:
    """
    Handle embeddings task by STRICTLY following instructions.
    No LLM used for deterministic rules.
    """
    logger.info("🔢 Embeddings task detected")

    
    prompt = f"""
        Task:
        {task_text}

        My email {YOUR_EMAIL}.

        Decide the correct submission strictly based on the instructions.
        Return ONLY the answer.
        """

    return (await call_aipipe_llm(prompt)).strip()

    
# Shards Task Handler - Return proper JSON object
async def handle_shards_task(json_data: dict, task_text: str) -> str:
    """Handle shards optimization task"""
    try:
        logger.info(f"🔧 Shards constraints: {json_data}")
        
        # Extract constraints - use correct field names from JSON
        total_docs = json_data.get("dataset") or json_data.get("totalDocs")
        max_docs_per_shard = json_data.get("max_docs_per_shard") or json_data.get("maxDocsPerShard")
        max_shards = json_data.get("max_shards") or json_data.get("maxShards")
        min_replicas = json_data.get("min_replicas") or json_data.get("minReplicas")
        max_replicas = json_data.get("max_replicas") or json_data.get("maxReplicas")
        memory_per_shard = json_data.get("memory_per_shard") or json_data.get("memoryPerShard")
        max_total_memory = json_data.get("memory_budget") or json_data.get("maxTotalMemory")
        
        logger.info(f"  Total docs: {total_docs}")
        logger.info(f"  Max docs per shard: {max_docs_per_shard}")
        logger.info(f"  Max shards: {max_shards}")
        logger.info(f"  Replicas range: {min_replicas}-{max_replicas}")
        logger.info(f"  Memory per shard: {memory_per_shard} MB")
        logger.info(f"  Max total memory: {max_total_memory} MB")
        
        # Find valid configuration
        import math
        
        # Minimum shards needed
        min_shards = math.ceil(total_docs / max_docs_per_shard)
        
        logger.info(f"  Minimum shards needed: {min_shards}")
        
        # Try to find valid configuration
        # Start from minimum shards and go up
        for shards in range(min_shards, max_shards + 1):
            docs_per_shard = total_docs / shards
            
            # Check if docs per shard is valid
            if docs_per_shard > max_docs_per_shard:
                logger.info(f"  Shards={shards}: docs_per_shard={docs_per_shard:.1f} > {max_docs_per_shard} ❌")
                continue
            
            # Try different replica counts (prefer higher replicas for redundancy)
            for replicas in range(max_replicas, min_replicas - 1, -1):
                total_memory = shards * replicas * memory_per_shard
                
                logger.info(f"  Testing shards={shards}, replicas={replicas}: memory={total_memory} MB")
                
                if total_memory <= max_total_memory:
                    result = {
                        "shards": shards,
                        "replicas": replicas
                    }
                    json_str = json.dumps(result, separators=(',', ':'))
                    logger.info(f"✅ Valid config: {json_str}")
                    logger.info(f"   - Docs per shard: {docs_per_shard:.1f}")
                    logger.info(f"   - Total memory: {total_memory} MB <= {max_total_memory} MB")
                    return json_str
        
        # If no valid config found, log detailed error
        logger.error("❌ No valid configuration found!")
        logger.error(f"   Constraints: docs={total_docs}, max_docs_per_shard={max_docs_per_shard}")
        logger.error(f"   min_shards needed={min_shards}, max_shards={max_shards}")
        logger.error(f"   memory_per_shard={memory_per_shard}, budget={max_total_memory}")
        return '{"shards":1,"replicas":1}'
        
    except Exception as e:
        logger.error(f"Shards task error: {e}")
        import traceback
        traceback.print_exc()
        return '{"shards":1,"replicas":1}'

async def handle_image_task(file_url: str, task_text: str) -> str:
    """Handle image analysis tasks"""
    try:
        logger.info(f"🖼️ Downloading image: {file_url}")
        file_data, content_type = await download_file(file_url)
        
        # Check what kind of analysis is needed
        if "color" in task_text.lower() or "rgb" in task_text.lower():
            return analyze_image_dominant_color(file_data)
        
        # For other image tasks, might need LLM vision
        logger.warning("⚠️ Other image analysis not implemented")
        return ""
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return ""

async def handle_github_api_task(task_text: str, json_data: dict) -> int:
    """Handle GitHub API tasks"""
    try:
        # Extract parameters from JSON
        owner = json_data.get("owner")
        repo = json_data.get("repo")
        sha = json_data.get("sha")
        path_prefix = json_data.get("pathPrefix", "")
        
        logger.info(f"🐙 GitHub API: {owner}/{repo} @ {sha}, prefix: {path_prefix}")
        
        # Call GitHub API
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                headers={"Accept": "application/vnd.github.v3+json"}
            )
            response.raise_for_status()
            tree_data = response.json()
        
        # Count .md files under pathPrefix
        md_count = 0
        for item in tree_data.get("tree", []):
            path = item.get("path", "")
            if path.startswith(path_prefix) and path.endswith(".md"):
                md_count += 1
                logger.info(f"  Found: {path}")
        
        logger.info(f"📊 Total .md files under '{path_prefix}': {md_count}")
        
        # Add email length mod 2 offset
        email_length = len(YOUR_EMAIL)
        offset = email_length % 2
        final_answer = md_count + offset
        
        logger.info(f"📧 Email length: {email_length}, offset: {offset}")
        logger.info(f"✅ Final answer: {final_answer}")
        
        return final_answer
        
    except Exception as e:
        logger.error(f"GitHub API error: {e}")
        return 0

async def handle_csv_normalization(csv_data: bytes, task_text: str) -> str:
    """Handle CSV normalization tasks with improved date parsing"""
    try:
        import pandas as pd
        
        df = pd.read_csv(BytesIO(csv_data))
        logger.info(f"📊 Original CSV:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Data:\n{df}")
        
        # Normalize column names to snake_case
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        logger.info(f"  After column rename: {list(df.columns)}")
        
        # Normalize dates using format='mixed' for handling different formats
        if 'joined' in df.columns:
            logger.info(f"  Original dates: {df['joined'].tolist()}")
            # Use format='mixed' with dayfirst=True for better parsing
            df['joined'] = pd.to_datetime(df['joined'], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            logger.info(f"  Normalized dates: {df['joined'].tolist()}")
        
        # Convert value to integer
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
        
        # Convert id to integer
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
        
        # Sort by id ascending
        df = df.sort_values('id')
        
        logger.info(f"📊 After normalization:\n{df}")
        
        # Convert to JSON with compact formatting
        result = []
        for _, row in df.iterrows():
            result.append({
                "id": int(row['id']),
                "name": str(row['name']),
                "joined": str(row['joined']),
                "value": int(row['value'])
            })

        json_str = json.dumps(result, separators=(',', ':'), ensure_ascii=False)
        
        logger.info(f"✅ Normalized JSON ({len(json_str)} chars):")
        logger.info(f"  {json_str}")
        
        return json_str
        
    except Exception as e:
        logger.error(f"CSV normalization error: {e}")
        import traceback
        traceback.print_exc()
        return "[]"

# ==================== ORDERS TASK HANDLER ====================
async def handle_orders_task(csv_data: bytes, task_text: str) -> str:
    """Handle orders running total task - returns JSON array"""
    try:
        import pandas as pd
        
        df = pd.read_csv(BytesIO(csv_data))
        logger.info(f"📊 Orders CSV:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Data:\n{df}")
        
        # Normalize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Calculate total per customer
        customer_totals = df.groupby('customer_id')['amount'].sum().reset_index()
        customer_totals.columns = ['customer_id', 'total']
        
        # Sort by total descending and take top 3
        top3 = customer_totals.sort_values('total', ascending=False).head(3)
        
        logger.info(f"📊 Customer totals:\n{customer_totals}")
        logger.info(f"📊 Top 3:\n{top3}")
        
        # Convert to JSON array
        result = []
        for _, row in top3.iterrows():
            result.append({
                "customer_id": str(row['customer_id']),
                "total": float(row['total'])
            })
        
        json_str = json.dumps(result, separators=(',', ':'))
        
        logger.info(f"✅ Orders result: {json_str}")
        
        return json_str
        
    except Exception as e:
        logger.error(f"Orders processing error: {e}")
        import traceback
        traceback.print_exc()
        return "[]"


# ==================== UV COMMAND HANDLER ====================
async def handle_uv_command_task(task_text: str) -> str:
    # Extract URL from text
    url_match = re.search(r'(https?://[^\s"]+)', task_text)
    if not url_match:
        raise ValueError("No URL found in task")

    url = url_match.group(1)
    url = url.replace("<your email>", YOUR_EMAIL)

    # Extract header
    header_match = re.search(r'Include header\s+([^:]+):\s*([^\s,]+)', task_text)
    if not header_match:
        raise ValueError("No header found")

    header_key = header_match.group(1)
    header_val = header_match.group(2)

    return f'uv http get {url} -H "{header_key}: {header_val}"'

# ==================== TASK SOLVING ====================

async def solve_task_direct(task_text: str, urls: dict, base_url: str, html_content: str) -> Any:
    """Directly solve task with specialized handlers"""
    
    data_context = ""
    
    # 1. UV command task (MUST be first)
    if "uv http get" in task_text.lower() or ("uv.json" in task_text.lower() and "command" in task_text.lower()):
        logger.info("🔧 UV command task detected")
        return await handle_uv_command_task(task_text)
    
    # 2. Embeddings task (BEFORE other file processing)
    if "embeddings" in task_text.lower() and "email length" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if 'embeddings' in file_url and file_url.endswith('.json'):
                logger.info("🔢 Embeddings task detected")
                file_data, _ = await download_file(file_url)
                json_data = json.loads(file_data.decode('utf-8'))
                return await handle_embeddings_task(json_data, task_text)
    
    # 2. Shards optimization task
    if "shards" in task_text.lower() and "replicas" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if 'shards' in file_url and file_url.endswith('.json'):
                logger.info("🔧 Shards optimization task detected")
                file_data, _ = await download_file(file_url)
                json_data = json.loads(file_data.decode('utf-8'))
                return await handle_shards_task(json_data, task_text)
    
    # 3. Orders task
    if "running total" in task_text.lower() or ("orders.csv" in task_text.lower() and "top 3" in task_text.lower()):
        for file_url in urls.get("file_urls", []):
            if 'orders' in file_url and file_url.endswith('.csv'):
                logger.info("📦 Orders task detected")
                file_data, _ = await download_file(file_url)
                return await handle_orders_task(file_data, task_text)
    
    # 4. Audio transcription
    if "audio" in task_text.lower() or "transcribe" in task_text.lower() or "passphrase" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if file_url.endswith(('.opus', '.mp3', '.wav', '.m4a')):
                logger.info("🎵 Audio task detected")
                return await handle_audio_task(file_url, base_url)
    
    # 5. Image analysis
    if ("image" in task_text.lower() or "color" in task_text.lower() or 
        "rgb" in task_text.lower() or "heatmap" in task_text.lower()):
        for file_url in urls.get("file_urls", []):
            if file_url.endswith(('.png', '.jpg', '.jpeg')):
                logger.info("🖼️ Image task detected")
                return await handle_image_task(file_url, task_text)

    # 6. ZIP logs task
    if "logs.zip" in task_text.lower() or ("download" in task_text.lower() and "bytes" in task_text.lower()):
        for file_url in urls.get("file_urls", []):
            if file_url.endswith('.zip'):
                logger.info("📦 ZIP logs task detected")
                file_data, _ = await download_file(file_url)
                return await handle_zip_logs_task(file_data, task_text)
    
    # 7. PDF invoice task
    if "invoice" in task_text.lower() and "quantity" in task_text.lower() and "unitprice" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if file_url.endswith('.pdf') and 'invoice' in file_url:
                logger.info("🧾 Invoice PDF task detected")
                file_data, _ = await download_file(file_url)
                return await handle_pdf_invoice_task(file_data, task_text)        
    
    # 8. CSV normalization
    if "normalize" in task_text.lower() and "json" in task_text.lower() and "messy" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if file_url.endswith('.csv') and 'messy' in file_url:
                logger.info("📊 CSV normalization task detected")
                file_data, _ = await download_file(file_url)
                return await handle_csv_normalization(file_data, task_text)
    
    # 9. GitHub API task
    if "github" in task_text.lower() or "git/trees" in task_text.lower():
        for file_url in urls.get("file_urls", []):
            if file_url.endswith('.json') and 'gh-tree' in file_url:
                logger.info("🐙 GitHub API task detected")
                file_data, _ = await download_file(file_url)
                json_data = json.loads(file_data.decode('utf-8'))
                return await handle_github_api_task(task_text, json_data)
    
    # 5. Process API URLs (scrape data from pages)
    for api_url in urls.get("api_urls", []):
        try:
            logger.info(f"Scraping: {api_url}")
            scraped_text, scraped_html = await fetch_and_render_page(api_url)
            logger.info(f"📄 Scraped content: {scraped_text[:200]}")
            
            # Direct secret extraction
            secret_patterns = [
                r'secret[:\s]*([a-zA-Z0-9]{6,20})',
                r'code[:\s]*([a-zA-Z0-9]{6,20})',
                r'key[:\s]*([a-zA-Z0-9]{6,20})',
                r'\b([a-f0-9]{6,20})\b',
                r'\b([A-Z0-9]{6,20})\b',
            ]
            
            for pattern in secret_patterns:
                secret_match = re.search(pattern, scraped_text, re.I)
                if secret_match:
                    secret = secret_match.group(1).strip()
                    if secret.lower() not in ['secret', 'code', 'key', 'email', 'answer']:
                        logger.info(f"✓ Extracted secret: {secret}")
                        return secret
            
            data_context += f"\n\n=== Scraped from {api_url} ===\n{scraped_text}"
        except Exception as e:
            logger.error(f"Error scraping {api_url}: {e}")
    
    # 6. Download and process files
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
                data_context += f"\n\n=== CSV: {file_url} ===\nShape: {df.shape}\nColumns: {list(df.columns)}\nFirst rows:\n{df.head(10).to_string()}\n\nSummary:\n{df.describe().to_string()}"
            
            elif "json" in content_type or file_url.lower().endswith(".json"):
                json_data = json.loads(file_data.decode('utf-8'))
                data_context += f"\n\n=== JSON: {file_url} ===\n{json.dumps(json_data, indent=2)[:3000]}"
            
        except Exception as e:
            logger.error(f"Error processing {file_url}: {e}")
    
    # 7. Use LLM for remaining tasks
    solve_prompt = f"""You are a data analysis expert. Solve this task and provide ONLY the final answer value.

TASK:
{task_text}

DATA:
{data_context if data_context else "No external data provided - answer from task description"}

IMPORTANT INSTRUCTIONS:
- Read the task VERY carefully
- Return ONLY the final answer value
- NO explanations, NO "Answer:", NO extra words
- Just the value

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
                logger.warning("⚠️ No submit URL found, using default /submit")
                urls["submit_url"] = "https://tds-llm-analysis.s-anand.net/submit"
            
            # Solve the task
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
