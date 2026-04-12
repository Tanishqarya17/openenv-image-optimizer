from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/{full_path:path}")
async def catch_all_get(full_path: str):
    return {"status": "alive", "message": "OpenEnv Container is Running"}

@app.post("/{full_path:path}")
async def catch_all_post(full_path: str, request: Request):
    return {"status": "POST OK", "message": "Environment ready for inference"}