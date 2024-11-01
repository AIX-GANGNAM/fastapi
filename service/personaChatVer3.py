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
from database import get_persona_collection, redis_client
from personas import personas
import re
import json
from firebase_admin import firestore
from database import db
from langchain_ollama import OllamaLLM  # 새로운 import 문
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ollama 대신 OllamaLLM 사용
llm = OllamaLLM(
    model="swchoi1994/exaone3-7-q8_0-gguf:latest",
    base_url="http://192.168.0.119:11434",
    temperature=0.5
)

# GPT-4 모델 추가
gpt4_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

def calculate_importance_llama(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""시스템: 당신은 대화의 중요도를 평가하는 분석 시스템입니다. 
        오직 1에서 10 사이의 정수만 출력해야 합니다.
        
        평가 기준:
        1-3: 일상적인 대화, 인사, 가벼운 잡담
        4-6: 개인적 경험, 감정 공유, 일반적인 의견 교환
        7-8: 중요한 정보, 깊은 통찰, 강한 감정 표현
        9-10: 매우 중요한 결정, 핵심 정보, 강력한 감정적 순간

        규칙:
        1. 반드시 1에서 10 사이의 정수만 출력하세요
        2. 다른 텍스트나 설명을 추가하지 마세요
        3. 숫자 외의 모든 출력은 무시됩니다

        분석할 대화:
        "{content}"

        중요도 점수:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"content": content})
    
    try:
        importance = re.search(r'\b([1-9]|10)\b', result['text'])
        if importance:
            return int(importance.group())
        else:
            print(f"유효하지 않은 중요도 값. 기본값 5를 사용합니다.")
            return 5
    except (AttributeError, ValueError):
        print(f"중요도를 숫자로 변환할 수 없습니다: {result}. 기본값 5를 사용합니다.")
        return 5

def summarize_content(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""시스템: 당신은 대화 내용을 정확하고 간단히 요약하는 전문가입니다.

        요약 규칙:
        1. 최대 50자 이내로 요약하세요
        2. 핵심 내용과 감정만 포함하세요
        3. 불필요한 설명이나 부연 설명을 제외하세요
        4. 객관적이고 명확한 문장으로 작성하세요
        5. 다음 형식을 반드시 지키세요: [감정/태도] + 핵심 메시지

        예시:
        입력: "나는 정말 화가 나! 어제 친구가 약속을 어겼어. 세 시간이나 기다렸다고!"
        출력: [분노] 친구의 약속 불이행으로 3시간 대기

        입력: "오늘 날씨가 너무 좋아서 기분이 좋아. 공원에서 산책하면서 커피도 마셨어."
        출력: [긍정] 좋은 날씨에 공원 산책과 커피

        분석할 대화:
        "{content}"

        요약:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"content": content})
    summary = result['text'].strip()
    
    # 50자 제한 적용
    if len(summary) > 50:
        summary = summary[:47] + "..."
        
    return summary

def get_long_term_memory_tool(params):
    if isinstance(params, dict):
        # 이미 dict라면 바로 get_long_term_memory 함수 호출
        return get_long_term_memory(
            params['uid'],
            params['persona_name'],
            params['query'],
            params.get('limit', 3)
        )
    elif isinstance(params, str):
        try:
            # 개행 문자와 앞뒤 공백을 제거하여 JSON 문자열로 변환
            params = params.replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)
            
            # 필요한 필드가 존재하는지 확인
            if not all(k in params_dict for k in ['uid', 'persona_name', 'query']):
                return "Action Input에 필수 필드가 없습니다. 'uid', 'persona_name', 'query'가 포함된 JSON 형식으로 입력해주세요."
            
            # get_long_term_memory 함수 호출
            return get_long_term_memory(
                params_dict['uid'],
                params_dict['persona_name'],
                params_dict['query'],
                params_dict.get('limit', 3)
            )
        except json.JSONDecodeError as e:
            # JSON 디코딩 실패 시 에러 메시지 출력
            print(f"JSON 파싱 오류: {str(e)}")  # 디버깅을 위한 로그 출력
            return "Action Input이 올바른 JSON 형식이 아닙니다. 큰따옴표를 사용하여 JSON 형식으로 입력해주세요."
    else:
        # 잘못된 타입의 params가 입력된 경우
        return "잘못된 Action Input 타입입니다. dict 또는 JSON 형식의 문자열이어야 합니다."



    
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
    
    # 포맷팅된 문자열 반환
    return [
        f"[{m['timestamp']}] [{m['type']}] (중요도: {m['importance']}) {m['content']}"
        for m in memories
    ]

# 장기 기억 함수
def store_long_term_memory(uid, persona_name, memory):
    collection = get_persona_collection(uid, persona_name)
    embedding = embeddings.embed_query(memory)
    collection.add(
        documents=[memory],
        metadatas=[{"timestamp": datetime.now().isoformat()}],
        ids=[f"{uid}_{persona_name}_{datetime.now().isoformat()}"],
        embeddings=[embedding]
    )

def get_long_term_memory(uid, persona_name, query, limit=3):
    collection = get_persona_collection(uid, persona_name)
    embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )
    return results['documents'][0] if results['documents'] else []

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
        description="ChromaDB에서 장기 기억을 가져옵니다. Input은 'uid', 'persona_name', 'query', 그리고 'limit'을 int 포함한 JSON 형식의 문자열이어야 합니다."
    ),
      Tool(
        name="Short Term Memory",
        func=get_short_term_memory_tool,
        description="""Redis에서 시간대별 기억을 검색합니다. Input은 다음 형식의 JSON이어야 합니다:
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
    # 팔로워 firestore 추가하기
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
    selected_personas = [request.persona1, request.persona2]
    previous_response = request.topic  # 최초 주제를 `previous_response`로 설정

    # 페르소나 쌍 이름을 정의된 순서에 따라 정렬
    pair_name = sort_personas(request.persona1, request.persona2)  # 예: "Anger_Disgust"

    # 정렬된 페르소나 쌍을 사용하여 chat_ref 생성
    chat_ref = db.collection('personachat').document(request.uid).collection(pair_name)

    for i in range(request.rounds):
        for persona in selected_personas:
            persona_info = personas[persona]

            # 에이전트 호출에 필요한 입력값 정의
            inputs = {
                "input": previous_response,
                "persona1_name": request.persona1,
                "persona1_description": personas[request.persona1]["description"],
                "persona1_tone": personas[request.persona1]["tone"],
                "persona1_example": personas[request.persona1]["example"],
                "persona2_name": request.persona2,
                "persona2_description": personas[request.persona2]["description"],
                "persona2_tone": personas[request.persona2]["tone"],
                "persona2_example": personas[request.persona2]["example"],
                "current_persona_name": persona,
                "agent_scratchpad": "",
                "uid": request.uid,
                "topic": request.topic,
            }

            # 에이전트 호출
            response = agents[persona].invoke(inputs)

            # 현재 페르소나에 따라 speaker 값을 설정
            speaker = persona  # 현재 페르소나를 speaker로 설정
            chat_ref.add({
                'speaker': speaker,
                'text': response['output'],
                'timestamp': datetime.now().isoformat(),
                'isRead': False
            })

            # 단기 기억에 저장
            store_short_term_memory(request.uid, persona, f"{persona}: {response['output']}")

            # 중요도 계산
            importance = calculate_importance_llama(response['output'])

            # 중요도가 8 이상이면 벡터 db에 저장
            if importance >= 5:
                store_long_term_memory(request.uid, persona, response['output'])

            # 현재 페르소나의 응답을 다음 입력으로 설정
            previous_response = response['output']

    return {"message": "Conversation simulated successfully."}  # 성공적으로 완료된 경우 반환        
        






# 대화 시뮬레이션 실행 예시
# chat_request = PersonaChatRequest(
#     uid="test01",
#     topic="안녕",
#     persona1="Anger",
#     persona2="Joy",
#     rounds=30
# )

# simulate_conversation(chat_request)







