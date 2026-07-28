FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY eval ./eval

# The index is built outside the image and mounted at runtime:
#   docker run -v $(pwd)/data:/app/data -p 8000:8000 finrag
ENV FINRAG_INDEX_DIR=/app/data/index

EXPOSE 8000
CMD ["uvicorn", "finrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
