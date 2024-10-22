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
        return get_long_term_memory(
            params['uid'],
            params['persona_name'],
            params['query'],
            params.get('limit', 3)
        )
    elif isinstance(params, str):
        # 문자열을 JSON으로 파싱
        try:
            params_dict = json.loads(params)
            return get_long_term_memory(
                params_dict['uid'],
                params_dict['persona_name'],
                params_dict['query'],
                params_dict.get('limit', 3)
            )
        except json.JSONDecodeError:
            return "Action Input이 올바른 JSON 형식이 아닙니다. 큰따옴표를 사용하여 JSON 형식으로 입력해주세요."
    else:
        return "잘못된 Action Input 타입입니다."

    
# 단기 기억 툴 정의
def get_short_term_memory_tool(params):
    if isinstance(params, dict):
        return get_short_term_memory(**params)
    elif isinstance(params, str):
        # 문자열을 JSON으로 파싱
        try:
            params_dict = json.loads(params)
            return get_short_term_memory(**params_dict)
        except json.JSONDecodeError:
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
        description="ChromaDB에서 장기 기억을 가져옵니다. Input은 'uid', 'persona_name', 'query', 그리고 선택���으로 'limit'을 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="Short Term Memory",
        func=get_short_term_memory_tool,
        description="Redis에서 단기 기억을 가져옵니다. Input은 'uid'와 'persona_name'을 포함한 JSON 형식의 문자열이어야 합니다."
    )
]

# 프롬프트 템플릿 정의
# Adjusted prompt template with uid
template = """Answer the following questions as best you can. You are currently acting as the persona named {persona_name}. Your uid is {uid}. Your persona has the following characteristics:

Description: {persona_description}
Tone: {persona_tone}
Example: {persona_example}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be a valid JSON string using double quotes.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Updated Tool function for Short Term Memory
Tool(
    name="Short Term Memory",
    func=lambda params: get_short_term_memory(**params) if isinstance(params, dict)
         else get_short_term_memory(*params.split(',')) if isinstance(params, str) else None,
    description="Retrieve short term memory from Redis. Input should be a JSON object with 'uid' and 'persona_name'. For example: {'uid': uid, 'persona_name': persona_name}"
)


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



async def simulate_conversation(request: PersonaChatRequest):
    selected_personas = [request.persona1, request.persona2]
    previous_response = request.topic  # 최초 주제를 `previous_response`로 설정

    # 고정된 chat_ref 생성
    chat_ref = db.collection('personachat').document(request.uid).collection(f"{request.persona1}_{request.persona2}")

    for i in range(request.rounds):
        for persona in selected_personas:
            persona_info = personas[persona]

            # 에이전트 호출에 필요한 입력값 정의
            inputs = {
                "input": previous_response,
                "persona_name": persona,
                "persona_description": persona_info["description"],
                "persona_tone": persona_info["tone"],
                "persona_example": persona_info["example"],
                "agent_scratchpad": "",
                "uid": request.uid,
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

            # 중요도가 8 이상이면 장기 기억에 저장
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

