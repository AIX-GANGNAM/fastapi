from pydantic import BaseModel, Field
from typing import List
import random

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
    target_name: str
    topic: str