FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required by Playwright browsers
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libatspi2.0-0 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libgbm1 \
    libxshmfence1 \
    libdrm2 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libxext6 \
    libxfixes3 \
    libwayland-egl1 \
    libwayland-client0 \
    libwayland-cursor0 \
    libgles2 \
    libegl1 \
    libopus0 \
    ffmpeg \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Remove playwright from requirements (will install separately)
RUN sed -i '/playwright/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir playwright

# Install browsers (Chromium + Firefox + Webkit)
RUN playwright install --with-deps chromium firefox webkit

# Add application
COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
