from fastapi import FastAPI, Request
import uvicorn
from func import handle_webhook

app = FastAPI()

@app.post("/capture-user-info")
async def capture_user_info(request: Request):
    user_input = await request.json()
    handle_webhook(user_input)

    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)