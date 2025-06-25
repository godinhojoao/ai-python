from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
import httpx

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is running on port 8000 at http://localhost:8000")


@app.get("/search")
async def search(q: str):
    url = f"https://www.google.com/search?q={q}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    return {
        "query": q,
        "status_code": resp.status_code,
        "content_length": len(resp.content),
        "content_snippet": resp.text[:100],
    }


# To run: uvicorn 5-simple-api:app --reload
# --reload makes the server automatically restart when you change your code.
# access http://localhost:8000/search?q=your-search-here
# access http://localhost:8000/search?q=testing
