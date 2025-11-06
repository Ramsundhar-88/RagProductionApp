To initalize
uv sync 
uv run uvicorn main:app --reload
npx inngest-cli@latest dev
uv run streamlit run streamlit_app.py

docker run -d \
  --name qdrantRagdb \
  -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
