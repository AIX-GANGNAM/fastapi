from pydantic import BaseModel, Field
from typing import List
import random

class ChatRequest(BaseModel):
    persona_name: str
    user_input: str
    user: dict

class ChatRequestV2(BaseModel):
    uid: str
    persona_name: str
    user_input: str

class ChatResponse(BaseModel):
    persona_name: str
    response: str

class FeedPost(BaseModel):
    id: str
    image: str
    caption: str
    likes: List[str] = []
    comments: List[dict] = []
    createdAt: str
    userId: str
    nick: str
    subCommentId: List[str] = []

class PersonaChatRequest(BaseModel):
    uid: str
    topic: str
    persona1: str
    persona2: str
    rounds: int

class ScheduleItem(BaseModel):
    time: str
    interaction_target: str
    topic: str
    conversation_rounds: int = Field(default_factory=lambda: random.randint(1, 4))

class PersonaSchedule(BaseModel):
    persona: str
    schedule: List[ScheduleItem]

class AllPersonasSchedule(BaseModel):
    schedules: List[PersonaSchedule]

class TaskRequest(BaseModel):
    uid: str
    persona_name: str
    interaction_target: str
    topic: str
    conversation_rounds: int
    time: str

# 요청으로 들어오는 데이터를 위한 Pydantic 모델
class SmsRequest(BaseModel):
    phone_number: str
    message: str  # 사용자 정의 메시지

class StarEventRequest(BaseModel):
    uid: str  # 사용자 ID
    eventId: str  # 이벤트 ID
    starred: bool  # 별표 상태
    time: str  # ISO 8601 형식의 시간
    userPhone: str  # 사용자 전화번호 추가
