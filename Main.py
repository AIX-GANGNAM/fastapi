from fastapi import FastAPI,  HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from models import ChatRequest, ChatResponse, FeedPost, PersonaChatRequest, TaskRequest, SmsRequest, StarEventRequest, ChatRequestV2, GeneratePersonalityRequest, UserProfile, CommentInteraction
import logging
from models import NotificationRequest
from service.sendNofiticaion import send_expo_push_notification
from service.services import (
    chat_with_persona,
    get_personas,
    create_feed_post,
    persona_chat,
    schedule_tasks,
    create_task,
    generate_and_save_user_schedule,
    get_user_schedule,
)
from typing import List
from datetime import datetime, timedelta
import pytz
from dateutil import parser
from database import db
from fastapi import Request   
from service.personaLoopChat import persona_chat_v2
from service.personaChatVer3 import simulate_conversation
from service.smsservice import send_sms_service
import uvicorn
from personaDebate import run_persona_debate

from service.personaGenerate import generate_personality
from service.profileUpdate import update_clone_personality
from service.interactionStore import store_user_interaction
from service.aiChatService import handle_offline_chat_service

from fastapi.middleware.cors import CORSMiddleware

# 스케줄러 초기화
scheduler = AsyncIOScheduler(
    timezone=pytz.timezone('Asia/Seoul'),
    job_defaults={
        'coalesce': True,  # 밀린 작업 중복 방지
        'max_instances': 5,  # 동시 실행 제한
        'misfire_grace_time': 300  # 5분까지 허용
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시
    try:
        scheduler.start()
        print("스케줄러가 시작되었습니다.")
    except Exception as e:
        print(f"스케줄러 시작 중 오류 발생: {str(e)}")
    yield
    # 애플리케이션 종료 시
    try:
        scheduler.shutdown()
        print("스케줄러가 종료되었습니다.")
    except Exception as e:
        print(f"스케줄러 종료 중 오류 발생: {str(e)}")

# FastAPI 앱 초기화
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 디바이스 - EndPoint 네트워크 통신 
@app.get("/v2/networkcheck")
async def network_check(request: Request):
    print("@app.get > network_check 호출")
    try:
        return {
            "status": "success",
            "message": "서버가 정상적으로 응답합니다",
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        logging.error(f"네트워크 체크 에러: {str(e)} | 요청: {request.url}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# 스케줄러 상태 확인 엔드포인트
@app.get("/scheduler-status")
async def get_scheduler_status():
    try:
        jobs = scheduler.get_jobs()
        return {
            "status": "running" if scheduler.running else "stopped",
            "jobs_count": len(jobs),
            "jobs": [
                {
                    "id": job.id,
                    "next_run_time": str(job.next_run_time)
                } for job in jobs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 라우트 정의
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    uid = chat_request.user.get('uid', '')
    
    response = await chat_with_persona(chat_request)
    
    # 딕셔너리에서 값을 추출합니다.
    persona_name = response['persona_name']
    response_text = response['response']
    
    
    # ChatResponse 모델에 맞게 반환
    return ChatResponse(persona_name=persona_name, response=response_text)
    # return await chat_with_persona(chat_request) 전에는 이거였음

@app.post("/v2/chat")
async def persona_chat_v2_endpoint(chat_request: ChatRequestV2):
    try:
        return await persona_chat_v2(chat_request)
    except Exception as e:
        logging.error(f"채팅 처리 에러: {str(e)} | 요청 데이터: {chat_request}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 오류: {str(e)}")


@app.get("/personas")
async def get_personas_endpoint():
    return get_personas()

@app.post("/feed") # 피드 생성 엔드포인트
async def create_feed_post_endpoint(post: FeedPost):
    print("@app.post /feed 호출")
    return await create_feed_post(post)

@app.post("/persona-chat") # 페르소나 상호간의 대화 테스트 엔드포인트
async def persona_chat_endpoint(chat_request: PersonaChatRequest):
    return await persona_chat(chat_request)

@app.post("/v3/persona-chat") # 이게 최신버전임
async def persona_chat_v3_endpoint(chat_request: PersonaChatRequest):
    return await simulate_conversation(chat_request)

@app.post("/execute-task") # 페르소나 상호간의 대화 테스트 엔드포인트
async def execute_task_endpoint(task_request: TaskRequest, background_tasks: BackgroundTasks):
    task = create_task(
        task_request.uid,
        task_request.persona_name,
        task_request.interaction_target,
        task_request.topic,
        task_request.conversation_rounds
    )
    background_tasks.add_task(task)
    return {"message": f"Task for {task_request.persona_name} interacting with {task_request.interaction_target} about {task_request.topic} at {task_request.time} has been scheduled."}

@app.post("/generate-user-schedule/{uid}")
async def generate_user_schedule_endpoint(uid: str, background_tasks: BackgroundTasks):
    all_schedules = generate_and_save_user_schedule(uid)
    background_tasks.add_task(schedule_tasks, uid, all_schedules)
    return {"message": f"Schedule generated and saved for user {uid}"}

@app.get("/user-schedule/{uid}")
async def get_user_schedule_endpoint(uid: str):
    schedule = get_user_schedule(uid)
    if schedule:
        return schedule
    raise HTTPException(status_code=404, detail="Schedule not found for this user")

@app.get("/networkcheck")
async def network_check_endpoint():
    print("network_check_endpoint 호출")
    return {"message": "Network check successful"}

# SMS 전송 엔드포인트 (Test 용)
@app.post("/send_sms")
def send_sms(request: SmsRequest):
    result = send_sms_service(request)  # 비동기로 서비스 함수 호출

    # 서비스 함수로부터 성공/실패 결과를 받아서 HTTPException 처리
    if result["status"] == "success":
        return {"message": result["message"]}
    else:
        raise HTTPException(status_code=result["status_code"], detail=result["message"])
    
@app.post("/star-event")
async def star_event_endpoint(request: StarEventRequest):
    if request.starred:
        try:
            kst = pytz.timezone('Asia/Seoul')
            
            # ISO 8601 시간 문자열을 파싱
            event_time = parser.parse(request.time)
            event_time_kst = event_time.astimezone(kst)
            scheduled_time = event_time_kst - timedelta(minutes=10)
            
            job_id = f"star_event_{request.eventId}"
            
            async def scheduled_task():
                try:
                    print(f"토론 시작 (KST): {datetime.now(kst)}")
                    result = await run_persona_debate(request)  # 토론 후 sms 보내는 함수
                    print(f"토론 완료 (KST): {datetime.now(kst)}")
                    return result
                except Exception as e:
                    print(f"토론 실행 중 오류: {str(e)}")
                    raise e
            
            # 기존 작업이 있다면 제거
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
            
            scheduler.add_job(
                scheduled_task,
                'date',
                run_date=scheduled_time,
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300
            )
            
            print(f"토론 예약됨 - ID: {job_id}")
            print(f"예약 시간 (KST): {scheduled_time}")
            return {"message": f"페르소나 토론이 {scheduled_time}에 예약되었습니다"}
            
        except Exception as e:
            print(f"스케줄 등록 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        job_id = f"star_event_{request.eventId}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            return {"message": "예약된 토론이 취소되었습니다"}
        return {"message": "취소할 토론이 없습니다"}
    

@app.post("/generate-personality")
async def generate_personality_endpoint(request: GeneratePersonalityRequest):
    return await generate_personality(request)


@app.post("/update-personality")
async def update_personality_endpoint(request: UserProfile):
    return await update_clone_personality(request)

@app.post("/store-comment-interaction")
async def store_comment_interaction(comment_data: CommentInteraction):
    """
    댓글 작성 시 사용자 상호작용을 저장하는 엔드포인트
    """
    return await store_user_interaction(
        uid=comment_data.uid,
        message=comment_data.content,
        interaction_type=comment_data.interaction_type
    )

@app.post("/clone-chat")
async def clone_chat_endpoint(chat_request: ChatRequest):
    return await handle_offline_chat_service(chat_request)

@app.post("/notification")
async def notification_endpoint(request: NotificationRequest):
    print("@app.post > notification_endpoint 호출") 
    print("request : ", request)
    return await send_expo_push_notification(request)

if __name__ == "__main__":
    print("FastAPI 서버 시작")
    uvicorn.run(app, host="0.0.0.0", port=8000)
