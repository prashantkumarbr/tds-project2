FROM mcr.microsoft.com/playwright/python:latest

WORKDIR /app

# Copy requirements EXCEPT playwright
COPY requirements.txt .

# Remove playwright from requirements (very important)
RUN sed -i '/playwright/d' requirements.txt

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
