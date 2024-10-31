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
    tone: str | None = None        # Optional
    example: str | None = None     # Optional
    description: str | None = None # Optional

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
    phone_number: str # ex) 01012345678
    message: str  # 사용자 정의 메시지

class StarEventRequest(BaseModel):
    uid: str  # 사용자 ID
    eventId: str  # 이벤트 ID
    starred: bool  # 별표 상태
    time: str  # ISO 8601 형식의 시간
    userPhone: str  # 사용자 전화번호 추가


class NotificationRequest(BaseModel):
    uid: str
    whoSendMessage: str
    message: str
    pushType: str


class GeneratePersonalityRequest(BaseModel):
    uid: str
    name: str
    personality: str
    speechStyle: str

class UserProfile(BaseModel):
    uid: str
    mbti: str
    personality: List[str]
    interests: List[str]
    communication_style: str
    speaking_style: str
    emoji_style: str
    values: List[str]
    decision_style: str

class CommentInteraction(BaseModel):
    uid: str
    content: str
    interaction_type: str

class ChatRequest(BaseModel):
    senderId: str          # 메시지 보낸 사용자 ID
    recipientId: str       # 수신자 ID
    chatId: str           # 채팅방 ID
    message: str          # 보낸 메시지

