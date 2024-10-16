from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, FeedPost, PersonaChatRequest
from services import (
    chat_with_persona,
    get_personas,
    create_feed_post,
    persona_chat,
)
from typing import List

app = FastAPI()

# 라우트 정의

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    return await chat_with_persona(chat_request)

@app.get("/personas")
async def get_personas_endpoint():
    return get_personas()

@app.post("/feed")
async def create_feed_post_endpoint(post: FeedPost):
    return await create_feed_post(post)

@app.post("/persona-chat")
async def persona_chat_endpoint(chat_request: PersonaChatRequest):
    return await persona_chat(chat_request)

if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버 실행")
    uvicorn.run(app, host="0.0.0.0", port=8000)