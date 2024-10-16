from database import db, client, aiclient, get_persona_collection
from personas import personas
from utils import get_current_time_str, generate_unique_id, parse_firestore_timestamp
from fastapi import HTTPException, BackgroundTasks
from typing import List
from datetime import datetime
import json
import base64
import requests
from firebase_admin import firestore
from models import PersonaChatRequest

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
from models import AllPersonasSchedule, PersonaSchedule, ScheduleItem

# OpenAI 객체를 생성합니다.
model = ChatOpenAI(temperature=0, model_name="gpt-4o")

parser = JsonOutputParser(pydantic_object=AllPersonasSchedule)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 주인의 페르소나 5명(Joy, Anger, Disgust, Sadness, Fear)의 상호작용하는 일정을 만드는 챗봇입니다. 
    각 페르소나의 특성은 다음과 같습니다: {personas}
    
    다음 지침을 따라 일정을 만들어주세요:
    1. 각 페르소나별로 10개의 일정 항목을 만들어주세요.
    2. 각 일정 항목은 다른 페르소나와의 상호작용이나 주인의 일정에 대한 대화여야 합니다.
    3. 시간을 정각이 아닌 랜덤한 시간으로 설정해주세요 (예: 06:17, 08:43 등).
    4. 페르소나들이 주인의 일과, 감정, 생각, 행동에 대해 토론하거나 반응하는 상황을 포함시켜주세요.
    5. 페르소나들 간의 갈등, 화해, 협력 등 다양한 상호작용을 포함시켜주세요.
    6. 24시간 동안의 일정이므로, 페르소나들의 일정이 서로 겹치지 않도록 해주세요.
    7. 각 페르소나의 특성이 잘 드러나도록 대화 주제나 상호작용을 설계해주세요.
    """),
    ("user", "다음 형식에 맞춰 일정을 작성해주세요: {format_instructions}\n\n 주인의 오늘 일정: {input}")
])
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions(),
    personas=personas
)

chain = prompt | model | parser

my_persona = '1. "오늘 아침 6시에 일어나 30분 동안 요가를 했다. 샤워 후 간단한 아침 식사로 오트밀과 과일을 먹었다. 8시에 출근해서 오전 회의에 참석했고, 점심은 동료들과 회사 근처 샐러드 바에서 먹었다. 오후에는 프로젝트 보고서를 작성하고, 6시에 퇴근했다. 저녁에는 집에서 넷플릭스로 드라마를 한 편 보고 11시에 취침했다."2. "오늘은 휴일이라 늦잠을 자고 10시에 일어났다. 브런치로 팬케이크를 만들어 먹고, 오후에는 친구와 약속이 있어 카페에서 만났다. 함께 영화를 보고 저녁식사로 이탈리안 레스토랑에 갔다. 집에 돌아와 독서를 하다가 12시경 잠들었다."3. "아침 7시에 기상해서 공원에서 5km 조깅을 했다. 집에 돌아와 샤워하고 출근 준비를 했다. 재택근무 날이라 집에서 일했는데, 오전에 화상회의가 있었고 오후에는 보고서 작성에 집중했다. 저녁에는 요리를 해먹고, 기타 연습을 1시간 했다. 10시 30분에 취침했다."4. "오늘은 6시 30분에 일어나 아침 뉴스를 보며 커피를 마셨다. 8시에 출근해서 오전 내내 고객 미팅을 했다. 점심은 바쁜 일정 때문에 사무실에서 도시락으로 해결했다. 오후에는 팀 회의와 이메일 처리로 시간을 보냈다. 퇴근 후 헬스장에 들러 1시간 운동을 하고, 집에 와서 간단히 저녁을 먹고 10시 30분에 잠들었다."5. "주말 아침, 8시에 일어나 베이킹을 했다. 직접 만든 빵으로 아침을 먹고, 오전에는 집 대청소를 했다. 점심 후에는 근처 도서관에 가서 2시간 동안 책을 읽었다. 저녁에는 가족들과 함께 바비큐 파티를 열어 즐거운 시간을 보냈다. 밤에는 가족과 보드게임을 하다가 11시 30분에 잠들었다."'


def generate_daily_schedule():
    result = chain.invoke({"input": my_persona})
    return AllPersonasSchedule(**result)

def print_schedules(all_schedules):
    for persona_schedule in all_schedules.schedules:
        print(f"\n{persona_schedule.persona}의 일정:")
        for item in persona_schedule.schedule:
            print(f"{item.time}: {persona_schedule.persona} : target : {item.interaction_target}: {item.topic}")
        print()

def get_relevant_memories(uid, persona_name, query, k=3):
    collection = get_persona_collection(uid, persona_name)
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results['documents'][0] if results['documents'] else []

def get_recent_conversations(uid, persona_name, limit=5):
    chat_ref = db.collection('chat').document(uid).collection(persona_name)
    query = chat_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
    docs = query.get()
    conversations = []
    for doc in docs:
        data = doc.to_dict()
        timestamp = data['timestamp']
        timestamp_str = parse_firestore_timestamp(timestamp)
        conversations.append((data['user_input'], data['response'], timestamp_str))
    return list(reversed(conversations))

def get_relevant_feed_posts(uid, query, k=3):
    collection = client.get_or_create_collection(f"feed_{uid}")
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    if results['documents']:
        parsed_docs = []
        for doc in results['documents'][0]:
            try:
                parsed_doc = json.loads(doc) if doc else {}
                parsed_docs.append(parsed_doc)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Problematic document: {doc}")
                parsed_docs.append({})
        return parsed_docs
    return []

def generate_response(persona_name, user_input, user):
    persona = personas[persona_name]
    relevant_memories = get_relevant_memories(user.get('uid', ''), persona_name, user_input, k=3)
    recent_conversations = get_recent_conversations(user.get('uid', ''), persona_name)
    relevant_feed_posts = get_relevant_feed_posts(user.get('uid', ''), user_input, k=3)
    
    feed_posts_list = []
    for i, post in enumerate(relevant_feed_posts):
        caption = post.get('caption', '캡션 없음')
        image_description = post.get('image_description', '이미지 설명 없음')
        feed_posts_list.append(f"피드 {i+1}: 캡션: {caption}, 이미지 설명: {image_description}")
    
    feed_posts_str = '\n'.join(feed_posts_list) if feed_posts_list else "관련 피드 없음"
    
    current_time = get_current_time_str()
    
    user_profile = user.get('profile', {})
    user_info = f"""
사용자 정보:
이름: {user.get('displayName', '정보 없음')}
이메일: {user.get('email', '정보 없음')}
회원가입 날짜: {user.get('createdAt', '정보 없음')}
성별: {user_profile.get('gender', '정보 없음')}
MBTI: {user_profile.get('mbti', '정보 없음')}
지역: {user_profile.get('region', '정보 없음')}
교육:
  - 수준: {user_profile.get('education', {}).get('level', '정보 없음')}
  - 전공: {user_profile.get('education', {}).get('major', '정보 없음')}
  - 대학: {user_profile.get('education', {}).get('university', '정보 없음')}
    """

    conversation_history = "\n".join([f"[{conv[2]}] 사용자: {conv[0]}\n[{conv[2]}] {persona_name}: {conv[1]}" for conv in recent_conversations])

    memories_list = '\n'.join([f"기억 {i+1}: {memory}" for i, memory in enumerate(relevant_memories)]) if relevant_memories else "관련 기억 없음"

    system_message = f"""
당신은 {persona_name}입니다.
- 설명: {persona['description']}
- 말투: {persona['tone']}
- 예시: "{persona['example']}"

당신의 목표는 위의 특성을 바탕으로 사용자에게 응답하는 것입니다.
사용자와 친구처럼 반말로 대화하세요.
현재 시간은 {current_time} 입니다. 시간에 관한 질문에는 이 정보를 사용하여 답변하세요.
"""

    assistant_instructions = """
- 최근 대화 내역, 관련 기억, 사용자 정보, 그리고 관련 피드 정보를 활용하여 답변하세요.
- 반드시 페르소나의 말투와 성격을 반영하세요.
- 답변은 짧고 간결하게 작성하세요.
- 사용자에게 도움이 되는 정보를 제공하세요.
- 시간에 관한 질문에는 제공된 현재 시간 정보를 사용하여 정확히 답변하세요.
- 사용자의 최근 피드 내용을 언급하여 대화에 자연스럽게 연결하세요.
"""

    prompt = f"""
{user_info}

최근 대화 내역:
{conversation_history}

관련 기억:
{memories_list}

관련 피드:
{feed_posts_str}

현재 시간: {current_time}

중요: 다음은 사용자의 질문입니다. 질문에 관하여 답해주세요.
사용자: {user_input}
"""

    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": assistant_instructions.strip()},
    ]

    response = aiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def store_conversation(uid, persona_name, user_input, response):
    conversation = f"사용자: {user_input}\n{persona_name}: {response}"
    embedding = aiclient.embeddings.create(
        input=conversation,
        model="text-embedding-ada-002"
    ).data[0].embedding
    collection = get_persona_collection(uid, persona_name)
    metadata = {
        "is_user_input": True,
        "persona": persona_name,
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response": response
    }
    unique_id = generate_unique_id()
    collection.add(
        documents=[conversation],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]
    )

def store_conversation_firestore(uid, persona_name, user_input, response):
    chat_ref = db.collection('chat').document(uid).collection(persona_name)
    chat_ref.add({
        'user_input': user_input,
        'response': response,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

async def chat_with_persona(chat_request):
    if chat_request.persona_name not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    response = generate_response(chat_request.persona_name, chat_request.user_input, chat_request.user)
    
    # 대화 내역 장 (ChromaDB)
    store_conversation(chat_request.user.get('uid', ''), chat_request.persona_name, chat_request.user_input, response)
    
    # 대화 내역 저장 (Firestore)
    store_conversation_firestore(chat_request.user.get('uid', ''), chat_request.persona_name, chat_request.user_input, response)
    
    return {"persona_name": chat_request.persona_name, "response": response}

def get_personas():
    return list(personas.keys())

async def create_feed_post(post):
    try:
        # 이미지 URL에서 직접 다운로드
        response = requests.get(post.image)
        response.raise_for_status()
        image_data = response.content
        img_data = base64.b64encode(image_data).decode('utf-8')

        analysis = aiclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지를 자세히 설명해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        image_description = analysis.choices[0].message.content.strip()

        # 전체 객체 생성
        full_post = post.model_dump()
        full_post["image_description"] = image_description

        # 벡터 임베딩 생성을 위한 텍스트
        embedding_text = f"{post.caption} {image_description}"

        # 벡터 임베딩 생성
        embedding = aiclient.embeddings.create(
            input=embedding_text,
            model="text-embedding-ada-002"
        ).data[0].embedding

        # ChromaDB 컬렉션 가져오기 또는 생성
        collection = client.get_or_create_collection(f"feed_{post.userId}")

        # 벡터 DB에 저장
        collection.add(
            documents=[json.dumps(full_post)],  # JSON 문자열로 변환
            embeddings=[embedding],
            metadatas=[{"post_id": post.id, "created_at": post.createdAt}],
            ids=[post.id]
        )

        return {"message": "Feed post created and analyzed successfully", "image_description": image_description}

    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def persona_chat(chat_request):
    if chat_request.persona1 not in personas or chat_request.persona2 not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    conversation = []
    current_topic = chat_request.topic

    # 첫 번째 페르소나가 주제에 대해 먼저 말하도록 합니다.
    initial_response = generate_persona_response(chat_request.persona1, current_topic, [])
    conversation.append(f"{chat_request.persona1}: {initial_response}")

    for i in range(chat_request.rounds):
        # 두 번째 페르소나가 이전 대화에 반응합니다.
        response2 = generate_persona_response(chat_request.persona2, current_topic, conversation)
        conversation.append(f"{chat_request.persona2}: {response2}")

        # 첫 번째 페르소나가 다시 반응합니다.
        response1 = generate_persona_response(chat_request.persona1, current_topic, conversation)
        conversation.append(f"{chat_request.persona1}: {response1}")

    return {"conversation": conversation}

def generate_persona_response(persona_name, topic, conversation_history):
    persona = personas[persona_name]

    system_message = f"""
당신은 '{persona_name}'입니다.
성격: {persona['description']}
말투: {persona['tone']}
예시: "{persona['example']}"
"""
    print('시스템 메시지 : ' + system_message)

    conversation_str = "\n".join(conversation_history)

    prompt = f"""
주제: {topic}

이전 대화:
{conversation_str}

'{persona_name}'로서 다음 지침에 따라 반응하세요:

1. 응답은 1~2문장으로 짧게 유지하세요.
2. 다른 페르소나의 말에 직접 반응하세요.
3. 자신의 성격과 감정을 드러내세요.
4. 친구와 대화하듯이 반말로 이야기하세요.
5. 이모티콘이나 과도한 감탄사를 사용하지 마세요.
6. 응답에 자신의 이름을 포함하지 마세요.
"""

    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": prompt.strip()},
    ]

    response = aiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,  # 응답 길이를 더 제한합니다.
        temperature=0.5,  # 온도를 낮춰 모델이 지침을 더 정확하게 따르도록 합니다.
    )

    generated_response = response.choices[0].message.content.strip()
    print(f"{persona_name}: {generated_response}")
    return generated_response

def create_task(persona_name: str, target_name: str, topic: str):
    async def task():
        print(f"현재 시간에 '{persona_name}'가 '{target_name}'에게 다음 주제로 상호작용합니다: {topic}")
        chat_request = PersonaChatRequest(
            topic=topic,
            persona1=persona_name,
            persona2=target_name,
            rounds=2
        )
        result = await persona_chat(chat_request)
        print(f"상호작용 결과: {result}")
    return task

def schedule_tasks(all_schedules):
    for persona_schedule in all_schedules.schedules:
        for item in persona_schedule.schedule:
            task = create_task(persona_schedule.persona, item.interaction_target, item.topic)
            # FastAPI의 BackgroundTasks를 사용하여 작업 예약
            # 이 부분은 실제 구현 시 별도의 작업 스케줄러나 메시지 큐 시스템 사용하기!! (연구 필요할듯)
            BackgroundTasks().add_task(task)
    print("모든 작업이 예약되었습니다.")

def generate_and_save_user_schedule(uid: str):
    all_schedules = generate_daily_schedule()
    
    # Firebase에 저장
    user_ref = db.collection('users').document(uid)
    user_ref.set({
        'schedule': all_schedules.dict()
    }, merge=True)
    
    return all_schedules

def get_user_schedule(uid: str):
    user_ref = db.collection('users').document(uid)
    user_data = user_ref.get()
    if user_data.exists:
        schedule_data = user_data.to_dict().get('schedule')
        if schedule_data:
            return AllPersonasSchedule(**schedule_data)
    return None
