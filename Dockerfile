FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fLo /usr/local/bin/tailwindcss https://github.com/tailwindlabs/tailwindcss/releases/download/v4.0.6/tailwindcss-linux-x64 \
    && chmod +x /usr/local/bin/tailwindcss

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN tailwindcss -i ./src/web/static/input.css -o ./src/web/static/output.css --minify

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

EXPOSE 8000

CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
