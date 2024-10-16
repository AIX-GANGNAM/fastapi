from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from models import ChatRequest, ChatResponse, FeedPost, PersonaChatRequest, TaskRequest
from services import (
    chat_with_persona,
    get_personas,
    create_feed_post,
    persona_chat,
    generate_daily_schedule,
    schedule_tasks,
    create_task,
)
from typing import List
from datetime import datetime

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    scheduler.add_job(update_daily_schedule, CronTrigger(hour=1, minute=0)) # 매일 새벽 1시에 스케쥴 초기화
    scheduler.start()
    yield
    # 종료 시 실행
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

async def update_daily_schedule():
    print("새로운 일정을 생성하고 등록합니다...")
    all_schedules = generate_daily_schedule()
    schedule_tasks(all_schedules)

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

@app.post("/execute-task")
async def execute_task_endpoint(task_request: TaskRequest, background_tasks: BackgroundTasks):
    task = create_task(task_request.persona_name, task_request.target_name, task_request.topic)
    background_tasks.add_task(task)
    return {"message": f"Task for {task_request.persona_name} interacting with {task_request.target_name} about {task_request.topic} has been scheduled."}

if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버 실행")
    uvicorn.run(app, host="0.0.0.0", port=8000)
