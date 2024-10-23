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
# Local Sllm API URL 설정
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"

# exaone-3.0-7.8b-instruct API로 중요도를 계산하는 함수
def calculate_importance_llama(content):
    prompt = f"""
다음 대화 내용의 중요성을 설명이나 추가 텍스트 없이 1에서 10까지 숫자로 평가해 주세요. 응답은 오직 숫자만 입력해주세요. 설명이나 추가 텍스트가 포함되면 응답은 무효로 처리됩니다. 반드시 1에서 10 사이의 정수만 반환해주세요.

대화 내용:
"{content}"
"""


    headers = {"Content-Type": "application/json"}
    data = {
        "model": "exaone-3.0-7.8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    response = requests.post(LLAMA_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        try:
            # 정규식을 사용하여 1~10 사이의 숫자만 추출
            importance = re.search(r'\b([1-9]|10)\b', result['choices'][0]['message']['content'])
            if importance:
                return int(importance.group())
            else:
                print(f"유효하지 않은 중요도 값. 기본값 5를 사용합니다.")
                return 5
        except (AttributeError, ValueError):
            print(f"중요도를 숫자로 변환할 수 없습니다: {result}. 기본값 5를 사용합니다.")
            return 5
    else:
        print(f"Llama API 호출 실패: {response.status_code} - {response.text}")
        return 5

# exaone-3.0-7.8b-instruct API로 요약하는 함수
def summarize_content(content):
    prompt = f"""
다음 대화 내용을 한두 문장으로 요약만 하세요. 반드시 요약만 입력하세요. 불필요한 설명이나 추가 텍스트는 절대 입력하지 마세요. 요약 외의 모든 응답은 무효로 처리됩니다. 반드시 짧고 간결하게 요약만 해주세요.

대화 내용:
"{content}"

요약:
"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "exaone-3.0-7.8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(LLAMA_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        return summary
    else:
        print(f"Llama API 호출 실패: {response.status_code} - {response.text}")
        return content  # 요약 실패 시 원본 반환

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
    if isinstance(params, dict):
        return get_short_term_memory(**params)
    elif isinstance(params, str):
        # 문자열을 JSON으로 파싱 (개행 문자와 기타 불필요한 문자 제거)
        try:
            # 개행 문자를 제거하고 앞뒤 공백을 제거하여 JSON으로 변환 가능하게 만듦
            params = params.replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)
            return get_short_term_memory(**params_dict)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {str(e)}")  # 디버깅을 위한 로그 출력
            return "Action Input이 올바른 JSON 형식이 아닙니다. 큰따옴표를 사용하여 JSON 형식으로 입력해주세요."
    else:
        return "잘못된 Action Input 타입입니다."


# 단기 기억 함수 (요약 및 대화 시간 포함)
def store_short_term_memory(uid, persona_name, memory):
    # 응답 요약
    summary = summarize_content(memory)
    
    # 현재 시간 추가
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 대화 내용과 시간을 함께 저장
    memory_with_time = f"[{current_time}] {summary}"
    
    # Redis에 저장
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    redis_client.lpush(redis_key, memory_with_time)
    redis_client.ltrim(redis_key, 0, 9)  # 단기 기억 10개만 유지

def get_short_term_memory(uid, persona_name):
    # Redis Key 출력
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    # Redis에서 데이터 가져오기
    chat_history = redis_client.lrange(redis_key, 0, 9)

    if not chat_history:
        print(f"No data found for {redis_key}")
        # decoded_history를 빈 리스트로 초기화
        decoded_history = []
    else:
        # 바이트 문자열을 디코딩하여 사람이 읽을 수 있는 형태로 변환
        decoded_history = [
            memory.decode('utf-8', errors='ignore') if isinstance(memory, bytes) else memory
            for memory in chat_history
        ]
        print(f"Data found for {redis_key}: {decoded_history}")

    # 디코딩된 데이터를 반환
    return decoded_history

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
    """
    사용자의 캘린더 이벤트를 Firestore에서 가져옵니다.
    
    :param params: JSON 형식의 문자열 또는 딕셔너리로 'uid'와 'date'를 포함해야 함
    :return: 사용자의 이벤트 목록 (리스트)
    """
    try:
        # params가 dict 형식이면 그대로 사용, 아니라면 문자열을 처리하여 변환
        if isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            # 이스케이프 문자가 포함된 경우 이를 제거
            params = params.replace("\\", "")
            # 개행 문자와 공백을 제거하여 JSON 문자열로 변환
            params = params.replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)
        
        # 필수 필드 'uid'와 'date'가 있는지 확인
        if not all(k in params_dict for k in ['uid', 'date']):
            return "Action Input에 필수 필드가 없습니다. 'uid'와 'date'가 포함된 JSON 형식으로 입력해주세요."
        
        uid = params_dict.get('uid')
        date = params_dict.get('date')

        # Firestore에서 사용자 문서를 가져옴
        user_ref = db.collection('calendar').document(uid)
        user_doc = user_ref.get()

        if user_doc.exists:
            events = user_doc.to_dict().get('events', [])  # 'events' 필드가 없으면 빈 리스트 반환
            
            # 특정 날짜의 이벤트만 필터링
            filtered_events = [event for event in events if event.get('date') == date]
            
            if not filtered_events:
                print(f"오늘은 사용자의 캘린더에 등록된 일정이 없습니다.")
                
            return filtered_events  # 필터링된 이벤트 반환
        else:
            return []  # 문서가 존재하지 않으면 빈 리스트 반환
    except json.JSONDecodeError as jde:
        print(f"JSON 파싱 오류: {str(jde)}")
        return "Action Input이 올바른 JSON 형식이 아닙니다."
    except Exception as e:
        print(f"Error fetching user events: {str(e)}")
        return []  # 오류 발생 시 빈 리스트 반환

def save_user_event(params):
    """
    사용자의 캘린더에 이벤트를 저장합니다.

    :param params: JSON 형식의 문자열 또는 딕셔너리로 'uid', 'date', 'time', 'title'을 포함해야 함
    :return: 저장 결과 메시지 (문자열)
    """
    try:
        # params가 dict 형식이면 그대로 사용, 아니라면 문자열을 처리하여 변환
        if isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            # 이스케이프 문자가 포함된 경우 이를 제거
            params = params.replace("\\", "")
            # 개행 문자와 공백을 제거하여 JSON 문자열로 변환
            params = params.replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)

        # 필수 필드 'uid', 'date', 'time', 'title'가 있는지 확인
        if not all(k in params_dict for k in ['uid', 'date', 'time', 'title']):
            return "Action Input에 필수 필드가 없습니다. 'uid', 'date', 'time', 'title'가 포함된 JSON 형식으로 입력해주세요."

        uid = params_dict.get('uid')
        date = params_dict.get('date')
        time = params_dict.get('time')
        title = params_dict.get('title')

        # Firestore에서 사용자 문서를 가져옴
        user_ref = db.collection('calendar').document(uid)
        user_doc = user_ref.get()

        # 기존 이벤트 목록에 새 이벤트 추가
        if user_doc.exists:
            events = user_doc.to_dict().get('events', [])
        else:
            events = []

        # 새 이벤트 생성
        new_event = {
            'date': date,
            'time': time,
            'title': title
        }

        # 이벤트 목록에 새 이벤트 추가
        events.append(new_event)

        # Firestore에 업데이트
        user_ref.set({'events': events}, merge=True)

        return f"이벤트가 성공적으로 저장되었습니다: {title} ({date} {time})"

    except json.JSONDecodeError as jde:
        print(f"JSON 파싱 오류: {str(jde)}")
        return "Action Input이 올바른 JSON 형식이 아닙니다."
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
        description="Redis에서 단기 기억을 가져옵니다. Input은 'uid'와 'persona_name'을 포함한 JSON 형식의 문자열이어야 합니다."
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
        description="user의 캘린더에 이벤트를 저장합니다. Input은 'uid', 'date', 'time', 'title'을 포함한 JSON 형식의 문자열이어야 합니다."
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

# LLM 모델 정의
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 페르소나별 에이전트 생성
agents = {}
# 프롬프트에 페르소나 설명, 톤, 예시를 넣도록 에이전트를 정의하는 부분 수정
for persona in personas:
    persona_info = personas[persona]
    
    search_agent = create_react_agent(
        llm, 
        tools, 
        ChatPromptTemplate.from_template(
            template,  # 템플릿 문자열만 전달
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
            if importance >= 8:
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

