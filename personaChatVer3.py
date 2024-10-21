from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

import requests
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel
from redis import Redis

from database import get_persona_collection, redis_client
from personas import personas

# LLAMA API URL 설정
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"

# LLAMA API로 중요도를 계산하는 함수
def calculate_importance_llama(content):
    prompt = f"""
다음 대화 내용의 중요성을 1에서 10까지 숫자로 평가해 주세요. 중요도는 다음 기준을 바탕으로 평가하세요:

1. 이 대화가 에이전트의 목표 달성에 얼마나 중요한가?
2. 이 대화가 에이전트의 감정이나 관계에 중요한 변화를 일으킬 수 있는가?
3. 이 대화가 에이전트의 장기적인 행동에 영향을 줄 수 있는가?

대화 내용:
"{content}"

응답은 오직 숫자만 입력해주세요. 설명이나 추가 텍스트 없이 1에서 10 사이의 정수만 반환해주세요.
"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    response = requests.post(LLAMA_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        try:
            importance = int(result['choices'][0]['message']['content'].strip())
            if 1 <= importance <= 10:
                return importance
            else:
                print(f"유효하지 않은 중요도 값: {importance}. 기본값 5를 사용합니다.")
                return 5
        except ValueError:
            print(f"중요도를 숫자로 변환할 수 없습니다: {result}. 기본값 5를 사용합니다.")
            return 5
    else:
        print(f"Llama API 호출 실패: {response.status_code} - {response.text}")
        return 5

# LLAMA API로 요약하는 함수
def summarize_content(content):
    prompt = f"""
다음 대화 내용을 한두 문장으로 간략하게 요약해 주세요.

대화 내용:
"{content}"

요약:
"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama",
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

# 툴 정의
web_search = TavilySearchResults(max_results=1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """Search the web by input keyword"""
    return web_search.invoke(query)

@tool
def get_current_time() -> str:
    """ALWAYS use this tool FIRST to get the current date and time before performing any task or search."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_long_term_memory(uid: str, persona_name: str, query: str, limit=3):
    """Get the long term memory from ChromaDB"""
    collection = get_persona_collection(uid, persona_name)
    embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )
    return results['documents'][0] if results['documents'] else []

# 단기 기억 함수 (요약 적용)
def store_short_term_memory(uid, persona_name, memory):
    # 응답 요약
    summary = summarize_content(memory)
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    redis_client.lpush(redis_key, summary)
    redis_client.ltrim(redis_key, 0, 9)  # 단기 기억 10개만 유지

def get_short_term_memory(uid, persona_name):
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    chat_history = redis_client.lrange(redis_key, 0, 9)
    return [memory.decode('utf-8') for memory in chat_history]

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

# 페르소나 정의
def create_persona_prompt(persona):
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {persona}. {personas[persona]['description']} "
                f"Your tone: {personas[persona]['tone']} "
                f"Example: {personas[persona]['example']} "
                "Always use the `get_current_time` tool first to ensure you have the most up-to-date information. "
                "Use the `search_web` tool only for the initial query to get live information. "
                "For subsequent interactions, rely on the chat history, your short-term memory, and long-term memory to continue the conversation. "
                "If you need to use the `get_long_term_memory` tool, please specify the UID and persona name."
            ),
            ("placeholder", "{chat_history}"),
            ("system", "Your short-term memory: {short_term_memory}"),
            ("system", "Your long-term memory: {long_term_memory}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [get_current_time, search_web, get_long_term_memory]

class LimitedWebSearchAgent(AgentExecutor, BaseModel):
    web_search_count: int = 0
    max_web_searches: int = 1  # 최대 웹 검색 횟수 설정

    def invoke(self, input, config=None):
        if self.web_search_count >= self.max_web_searches:
            # 웹 검색 도구를 제거
            self.tools = [tool for tool in self.tools if tool.name != "search_web"]
        result = super().invoke(input, config)
        if "search_web" in str(result):
            self.web_search_count += 1
        return result

agents = {}
agent_executors = {}  # 에이전트 실행기를 저장할 딕셔너리
for persona in personas:
    prompt = create_persona_prompt(persona)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = LimitedWebSearchAgent(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        max_execution_time=10,
        handle_parsing_errors=True,
    )
    agents[persona] = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    agent_executors[persona] = agent_executor  # 에이전트 실행기 저장

class PersonaChatRequest(BaseModel):
    uid: str
    topic: str
    persona1: str
    persona2: str
    rounds: int

def simulate_conversation(request: PersonaChatRequest):
    print(f"Topic: {request.topic}\n")
    selected_personas = [request.persona1, request.persona2]
    
    previous_response = request.topic  # 첫 번째 페르소나의 첫 입력은 주어진 주제로 설정

    for i in range(request.rounds):
        for persona in selected_personas:
            # Redis에서 채팅 기록 불러오기
            chat_history = get_short_term_memory(request.uid, persona)
            print(f"chat_history for {persona}: {chat_history}")       
            # 장기 기억 검색 (현재 대화 내용을 기반으로)
            long_term_memories = get_long_term_memory(request.uid, persona, previous_response)
            # 이전 페르소나의 응답을 현재 페르소나의 입력으로 사용
            response = agents[persona].invoke(
                {
                    "input": previous_response,  # 이전 페르소나의 응답을 기반으로 대화를 이어감
                    "chat_history": chat_history,  # Redis에서 불러온 단기 기억 전달
                    "short_term_memory": "\n".join(chat_history),
                    "long_term_memory": "\n".join(long_term_memories),
                },
                {"configurable": {"session_id": f"{request.uid}_{persona}_session"}}
            )
            print(f"{persona}: {response['output']}\n")
            
            # 새로운 대화 내용을 요약하여 단기 기억에 저장
            store_short_term_memory(request.uid, persona, f"{persona}: {response['output']}")
            store_short_term_memory(request.uid, persona, f"상대방: {previous_response}")

            # 중요도 계산
            importance = calculate_importance_llama(response['output'])
            print(f"중요도: {importance}")

            # 중요도가 8 이상이면 장기 기억에 저장
            if importance >= 8:
                store_long_term_memory(request.uid, persona, response['output'])
                print(f"{persona}의 응답이 장기 기억에 저장되었습니다.")

            # 현재 페르소나의 응답을 다음 페르소나의 입력으로 설정
            previous_response = response['output']
        
        print("---")
    
    # 대화 종료 후 웹 검색 카운트 초기화
    for persona in selected_personas:
        agent_executors[persona].web_search_count = 0  # web_search_count 직접 초기화

# 대화 시뮬레이션 실행 예시
chat_request = PersonaChatRequest(
    uid="user123",
    topic="오늘 날씨에 관해서 얘기하자",
    persona1="Joy",
    persona2="Anger",
    rounds=3
)

simulate_conversation(chat_request)
