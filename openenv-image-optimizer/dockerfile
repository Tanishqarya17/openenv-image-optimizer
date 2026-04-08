FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your scripts (including inference.py for check #3)
COPY --chown=user . .

# Run the FastAPI server on port 7860 to accept the POST checks
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
