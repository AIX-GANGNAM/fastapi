from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    persona_name: str
    user_input: str
    user: dict

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
    topic: str
    persona1: str
    persona2: str
    rounds: int = 3  # 기본적으로 3번의 대화 주고받기

class ScheduleItem(BaseModel):
    time: str
    interaction_target: str
    topic: str

class PersonaSchedule(BaseModel):
    persona: str
    schedule: List[ScheduleItem]

class AllPersonasSchedule(BaseModel):
    schedules: List[PersonaSchedule]

class TaskRequest(BaseModel):
    persona_name: str
    target_name: str
    topic: str