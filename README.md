# SHL Recommender

FastAPI service that recommends SHL assessments using retrieval and optional LLM assistance.

## Architecture

```
[Client]
   |
   v
[FastAPI /chat]
   |
   v
[RecommenderAgent] ---> [Guardrails]
   |
   v
[CatalogRetriever] ---> [Catalog JSON]
   |
   v
[Optional Gemini LLM]
```

## Setup

1. Clone and install dependencies:

```bash
git clone <your-repo-url>
cd shl-recommender
python -m venv .venv
. .venv/bin/activate
# On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create environment file:

```bash
cp .env.example .env
```

3. Scrape the catalog (optional but recommended):

```bash
python scripts/scrape_catalog.py --output data/catalog.json
```

4. Run the API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a mid-level Java developer with strong communication skills"}
    ]
  }'
```

## Tests

```bash
pytest
```

## Deployment (Render.com)

1. Create a new Web Service in Render.
2. Connect the repository.
3. Use the included render.yaml or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add the GEMINI_API_KEY env var (optional).
5. Deploy.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| GEMINI_API_KEY | Google Gemini API key (optional) | empty |
| PORT | Server port | 8000 |
| LOG_LEVEL | Logging level | INFO |
| CATALOG_PATH | Path to catalog JSON | data/catalog.json |
| MAX_TURNS | Max user/assistant turns | 8 |
| TOP_K_RETRIEVAL | Number of retrieved items | 15 |
