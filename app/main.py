from fastapi import FastAPI

app = FastAPI(title="Closer CRM API")

@app.get("/health")
def health():
    return {"status": "ok"}
