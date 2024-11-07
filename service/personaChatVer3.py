from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

import os
import requests
from typing import List, Dict
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel
from redis import Redis
from database import redis_client, get_user_collection, query_memories
from personas import personas
import re
import json
from firebase_admin import firestore
from database import db
from langchain_ollama import OllamaLLM  # 새로운 import 문
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import asyncio
from fastapi import HTTPException

# Ollama 대신 OllamaLLM 사용
llm = OllamaLLM(
    model="swchoi1994/exaone3-7-q8_0-gguf:latest",
    base_url="http://192.168.0.119:11434",
    temperature=0.5
)

# GPT-4 모델 추가
gpt4_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

async def calculate_importance_llama(text: str) -> int:
    """텍스트의 중요도를 계산"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = PromptTemplate.from_template("""
        다음 텍스트의 중요도를 1-10 사이의 숫자로 평가해주세요.
        평가 기준:
        - 감정적 강도
        - 정보의 가치
        - 기억할 필요성
        
        텍스트: {text}
        
        중요도 (1-10):""")
        
        # 체인 실행
        chain = prompt | llm
        result = await chain.ainvoke({"text": text})
        importance = int(result.content.strip())
        
        return max(1, min(10, importance))  # 1-10 사이로 제한
        
    except Exception as e:
        print(f"중요도 계산 중 오류: {str(e)}")
        return 5  # 오류 발생시 기본값

async def summarize_content(text: str) -> str:
    """텍스트 요약"""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = PromptTemplate.from_template("""
        다음 텍스트를 50자 이내로 핵심만 간단히 요약해주세요:
        
        텍스트: {text}
        
        요약:""")
        
        # 새로운 방식으로 체인 구성
        chain = prompt | llm
        
        # 체인 실행
        result = await chain.ainvoke({"text": text})
        return result.content.strip()
        
    except Exception as e:
        print(f"요약 중 오류: {str(e)}")
        return text[:50] + "..."  # 오류 발생시 단순 절삭

def get_long_term_memory_tool(params):
    """벡터 DB에서 메모리 검색 도구"""
    try:
        if isinstance(params, dict):
            params_dict = params
        else:
            params = params.replace('\\"', '"').strip('"')
            params_dict = json.loads(params)
        
        results = query_memories(
            uid=params_dict.get('uid'),
            query=params_dict.get('query'),
            memory_type=params_dict.get('type'),
            persona_name=params_dict.get('persona_name'),
            limit=params_dict.get('limit', 3)
        )
        
        # 결과 포맷팅
        formatted_memories = []
        for result in results:
            content = result['content']
            metadata = result['metadata']
            formatted_memories.append(
                f"[{metadata['timestamp']}] "
                f"(중요도: {metadata['importance']}) "
                f"{content.get('message', content) if isinstance(content, dict) else content}"
            )
        
        return formatted_memories
        
    except Exception as e:
        print(f"Error in get_long_term_memory_tool: {str(e)}")
        return f"메모리 검색 중 오류 발생: {str(e)}"



    
# 단기 기억 툴 정의 (개선된 JSON 파싱 로직 추가)
def get_short_term_memory_tool(params):
    try:
        if isinstance(params, dict):
            params_dict = params
        else:
            # 문자열에서 이스케이프된 따옴표 처리
            params = params.replace('\\"', '"')
            # 앞뒤의 따옴표 제거
            params = params.strip('"')
            params_dict = json.loads(params)
        
        return get_short_term_memory(
            uid=params_dict.get('uid'),
            persona_name=params_dict.get('persona_name')
        )
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {str(e)}")
        return "JSON 파싱 오류가 발생했습니다."
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return f"오류가 발생했습니다: {str(e)}"


# 단기 기억 함수 (요약 및 대화 시간 포함)
def store_short_term_memory(uid, persona_name, memory):
    # 응답 요약
    summary = summarize_content(memory)
    
    # 현재 시간 추가
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 메모리 데이터 구조화
    memory_data = {
        "timestamp": timestamp,
        "content": summary,
        "importance": calculate_importance_llama(memory),
        "type": "chat"  # 'chat', 'event', 'emotion' 등으로 구분 가능
    }
    
    # JSON으로 직렬화
    memory_json = json.dumps(memory_data, ensure_ascii=False)
    
    # Redis 키 설정
    base_key = f"{uid}:{persona_name}"
    
    # 시간대별 저장
    time_keys = {
        "recent": {
            "key": f"{base_key}:recent",
            "max_items": 20,
            "ttl": 3600  # 1시간
        },
        "today": {
            "key": f"{base_key}:today",
            "max_items": 50,
            "ttl": 86400  # 24시간
        },
        "weekly": {
            "key": f"{base_key}:weekly",
            "max_items": 100,
            "ttl": 604800  # 1주일
        }
    }
    
    # 각 시간대별로 저장
    for storage_type, config in time_keys.items():
        # 중요도가 7 이상인 경우만 weekly에 저장
        if storage_type == "weekly" and memory_data["importance"] < 7:
            continue
            
        redis_client.lpush(config["key"], memory_json)
        redis_client.ltrim(config["key"], 0, config["max_items"] - 1)
        redis_client.expire(config["key"], config["ttl"])

def get_short_term_memory(uid, persona_name, memory_type="recent"):
    base_key = f"{uid}:{persona_name}:{memory_type}"
    
    # Redis에서 데이터 가져오기
    raw_memories = redis_client.lrange(base_key, 0, -1)
    
    if not raw_memories:
        return []
        
    # JSON 디코딩 및 시간순 정렬
    memories = []
    for memory in raw_memories:
        try:
            decoded = json.loads(memory)
            memories.append(decoded)
        except json.JSONDecodeError:
            continue
            
    # 시간순 정렬
    memories.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))
    
    # 포맷팅된 문자 반환
    return [
        f"[{m['timestamp']}] [{m['type']}] (중요도: {m['importance']}) {m['content']}"
        for m in memories
    ]

# 장기 기억 함수
def store_long_term_memory(uid: str, persona_name: str, memory: str, memory_type: str):
    """벡터 DB에 통합 메모리 저장"""
    collection = get_user_collection(uid)
    
    # 임베딩 생성
    embedding = gpt4_model.embeddings.create(
        input=memory,
        model="text-embedding-ada-002"
    ).data[0].embedding
    
    # 메타데이터 구성
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "type": memory_type,  # 파라미터로 받은 타입 사용
        "persona_name": persona_name,
        "importance": calculate_importance_llama(memory)
    }
    
    # 고유 ID 생성
    unique_id = f"{uid}_{metadata['type']}_{metadata['persona_name']}_{metadata['timestamp']}"
    
    # 컬렉션에 저장
    collection.add(
        documents=[memory],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]
    )

def get_long_term_memory(uid, persona_name, query, limit=3):
    """벡터 DB에서 페르소나 관련 메모리 검색"""
    collection = get_user_collection(uid)
    embedding = embeddings.embed_query(query)
    
    # 검색 실행
    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit,
        where={"type": "persona_chat", "persona_name": persona_name}
    )
    
    # 결과�� 있는 경우 content 필드를 반환
    if results['metadatas'] and results['documents']:
        formatted_memories = []
        for metadata, document in zip(results['metadatas'][0], results['documents'][0]):
            timestamp = metadata.get('timestamp', '')
            importance = metadata.get('importance', 0)
            content = metadata.get('content', document)  # content가 없으면 document 사용
            formatted_memories.append(f"[{timestamp}] (중요도: {importance}) {content}")
        return formatted_memories
    return []

def get_user_profile(params):
    try:
        # uid 값을 추출합니다.
        if isinstance(params, str):
            params = json.loads(params)
        uid = params.get('uid')
        
        # Firestore의 'users/{UID}' 경로에서 프로필 필드를 가져옵니다.
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()

        if user_doc.exists:
            profile = user_doc.to_dict().get('profile')
            if profile:
                return profile
            else:
                return f"유저 {uid}의 프로필을 찾을 수 없습니다."
        else:
            return f"유저 {uid}의 문서를 찾을 수 없습니다."
    except Exception as e:
        # 예외가 발생한 경우 에러 메시지를 반환합니다.
        return f"Firestore에서 유저 프로필을 가져오는 중 오류가 발생했습니다: {str(e)}"



def get_user_events(params):
    try:
        if isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            params = params.replace("\\", "").replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)
        
        if not all(k in params_dict for k in ['uid', 'date']):
            return "Action Input에 필수 필드가 없습니다."
        
        uid = params_dict.get('uid')
        date = params_dict.get('date')

        user_ref = db.collection('calendar').document(uid)
        user_doc = user_ref.get()

        if user_doc.exists:
            events = user_doc.to_dict().get('events', [])
            
            filtered_events = [
                {
                    'date': event.get('date'),
                    'time': event.get('time').strftime('%Y년 %m월 %d일 %p %I시 %M분 %S초 UTC%z') if isinstance(event.get('time'), datetime) else str(event.get('time')),
                    'title': event.get('title')
                }
                for event in events if event.get('date') == date
            ]
            
            if not filtered_events:
                print(f"오늘은 사용자의 캘린더에 등록된 일정이 없습니다.")
                
            return filtered_events
        else:
            return []

    except Exception as e:
        print(f"Error fetching user events: {str(e)}")
        return []

def save_user_event(params):
    try:
        if isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            params = params.replace("\\", "").replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)

        if not all(k in params_dict for k in ['uid', 'date', 'timestamp', 'title']):
            return "Action Input에 필수 필드가 없습니다."

        uid = params_dict.get('uid')
        date = params_dict.get('date')  # 날짜 문자열 (예: "2024-10-24")
        time_str = params_dict.get('timestamp')  # 시간 문자열 (예: "12:30:00")
        title = params_dict.get('title')

        # timestamp를 datetime 객체로 변환
        full_datetime_str = f"{date}T{time_str}+09:00"  # ISO 형식으로 변환
        timestamp = datetime.fromisoformat(full_datetime_str)

        # Firestore에 저장할 데이터 형식
        new_event = {
            'date': date,  # 문자열 형식 (예: "2024-10-24")
            'time': timestamp,  # Timestamp 객체
            'title': title,  # 문자열
            'starred': False  # 기본값
        }

        # Firestore에 저장
        user_ref = db.collection('calendar').document(uid)
        user_doc = user_ref.get()

        events = user_doc.to_dict().get('events', []) if user_doc.exists else []
        events.append(new_event)
        
        user_ref.set({'events': events}, merge=True)

        return f"이벤트가 성공적으로 저장되었습니다: {title}"

    except Exception as e:
        print(f"Error saving user event: {str(e)}")
        return f"이벤트 저장 중 오류가 발생했습니다: {str(e)}"

# 툴 정의
web_search = TavilySearchResults(max_results=1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

tools = [
    Tool(
        name="Search",
        func=web_search.invoke,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Current Time",
        func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 인수를 받도록 수정
        description="ALWAYS use this tool FIRST to get the current date and time before performing any task or search."
    ),
     Tool(
        name="Long Term Memory",
        func=get_long_term_memory_tool,
        description="""ChromaDB에서 기억을 검색합니다. Input은 다음 형식의 JSON이어야 합니다:
        {
            "uid": "사용자ID",
            "query": "검색할 내용",
            "limit": 검색 결과 개수 (선택, 기본값: 3),
            "type": "검색할 메모리 타입" (선택, 생략 가능)
        }
        
        type 옵션:
        - 생략시: 모든 타입의 메모리 검색
        - "persona_chat": 페르소나 채팅 메모리만 검색
        - "event": 이벤트 메모리만 검색
        - "emotion": 감정 메모리만 검색
        - "clone": 사용자 분신 채팅 메모리만 검색
        
        반환 형식: [시간] (타입: X) 내용"""
    ),
      Tool(
        name="Short Term Memory",
        func=get_short_term_memory_tool,
        description="""Redis에서 시간대별 기억 검색합니다. Input은 다음 형식의 JSON이어야 합니다:
        {
            "uid": "사용자ID",
            "persona_name": "페르소나이름",
            "memory_type": "recent/today/weekly" (선택, 기본값: recent)
        }
        
        memory_type 설명:
        - recent: 최근 1시간 내 기억 (최대 20개)
        - today: 오늘의 기억 (최대 50개)
        - weekly: 일주일 내 중요 기억 (최대 100개, 중요도 7 이상)
        
        반환 형식: [시간] [타입] (중요도: X) 내용"""
    ),
    Tool(
        name="Search Firestore for user profile",
        func=get_user_profile,
        description="Firestore에서 유저 프로필을 검색합니다. Input은 'uid'를 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="owner's calendar",
        func=get_user_events,
        description="user의 캘린더를 가져옵니다. Input은 'uid'와 'date'를 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="save user event",
        func=save_user_event,
        description="user의 캘린더에 이벤트를 저장합니다. Input은 'uid', 'date', 'timestamp', 'title'을 포함한 JSON 형식의 문자열이어야 합니다."
    )
    # 팔워 firestore 추가하기
]

# 프롬프트 템플릿 정의
# Adjusted prompt template with uid
template = """
You are currently acting as two personas. Below are the details for each persona:

Persona 1:
- Name: {persona1_name}
- Description: {persona1_description}
- Tone: {persona1_tone}
- Example dialogue: {persona1_example}

Persona 2:
- Name: {persona2_name}
- Description: {persona2_description}
- Tone: {persona2_tone}
- Example dialogue: {persona2_example}

Owner's UID: {uid}

You both need to discuss the following topic provided by the user: "{topic}". 
You will take turns responding in the conversation, and you should acknowledge what the other persona has said.

It is now {current_persona_name}'s turn.

You have access to the following tools:
{tools}

Use the following format for each response:

Question: the input question or topic to discuss
Thought: think about what to say or do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be a valid JSON string using double quotes.
Observation: the result of the action
... (This Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: provide the final answer or response

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

# 페르소나별 에이전트 생성
agents = {}
# 프롬프트에 페르소나 설명, 톤, 예시를 넣도록 에이전트를 정의하는 부분 수정
for persona in personas:
    persona_info = personas[persona]
    
    search_agent = create_react_agent(
        gpt4_model,  # GPT-4 모델 사용
        tools, 
        ChatPromptTemplate.from_template(
            template,
        )
    )
    
    agents[persona] = AgentExecutor(
        agent=search_agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
class PersonaChatRequest(BaseModel):
    uid: str
    topic: str
    persona1: str
    persona2: str
    rounds: int


PERSONA_ORDER = ['Joy', 'Anger', 'Disgust', 'Sadness', 'Fear']

def sort_personas(persona1, persona2):
    """정의된 순서에 따라 페르소나를 정렬하여 쌍 이름을 생성합니다."""
    index1 = PERSONA_ORDER.index(persona1)
    index2 = PERSONA_ORDER.index(persona2)
    if index1 < index2:
        return f"{persona1}_{persona2}"
    else:
        return f"{persona2}_{persona1}"

async def simulate_conversation(request: PersonaChatRequest):
    try:
        selected_personas = [request.persona1, request.persona2]
        pair_name = sort_personas(request.persona1, request.persona2)
        chat_ref = db.collection('personachat').document(request.uid).collection(pair_name)
        
        previous_response = request.topic
        
        for i in range(request.rounds):
            for persona in selected_personas:
                try:
                    # ... 기존 대화 생성 코드 ...
                    
                    # 메모리 저장 부분을 비동기로 처리
                    asyncio.create_task(store_conversation_memory(
                        request.uid,
                        persona,
                        response['output'],
                        chat_ref
                    ))
                    
                    previous_response = response['output']
                    
                except Exception as e:
                    print(f"대화 라운드 처리 오류: {str(e)}")
                    continue
                    
        return {"message": "대화가 성공적으로 완료되었습니다."}
        
    except Exception as e:
        print(f"대화 시뮬레이션 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def store_conversation_memory(uid, persona, message, chat_ref):
    try:
        # Firestore에 저장
        chat_ref.add({
            'speaker': persona,
            'text': message,
            'timestamp': datetime.now().isoformat(),
            'isRead': False
        })
        
        # 단기 메모리 저장
        store_short_term_memory(uid, persona, f"{persona}: {message}")
        
        # 중요도 계산
        importance = await calculate_importance_llama(message)
        
        # 중요도가 5 이상이면 장기 메모리 저장
        if importance >= 5:
            store_long_term_memory(
                uid=uid,
                persona_name=persona,
                memory=message,
                memory_type="conversation"
            )
            
    except Exception as e:
        print(f"대화 메모리 저장 오류: {str(e)}")






# 대화 시뮬레이션 실행 예시
# chat_request = PersonaChatRequest(
#     uid="test01",
#     topic="안녕",
#     persona1="Anger",
#     persona2="Joy",
#     rounds=30
# )

# simulate_conversation(chat_request)







