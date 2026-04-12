from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.get("/{full_path:path}")
async def catch_all_get(full_path: str):
    return {"status": "alive", "message": "OpenEnv Container is Running"}

@app.post("/{full_path:path}")
async def catch_all_post(full_path: str, request: Request):
    return {"status": "POST OK", "message": "Environment ready for inference"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()