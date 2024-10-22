import os
from dotenv import load_dotenv
import chromadb
from database import redis_client, db, get_persona_collection
from datetime import datetime
from typing import List
import requests
from fastapi import HTTPException
from personas import personas
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("persona_chat_v2")
# 환경 변수 로드
load_dotenv()

# OpenAI 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Tavily 검색 도구 초기화
web_search = TavilySearchResults(max_results=1)

# Llama API URL 설정
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"

# Llama API로 중요도를 계산하는 함수
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
            print(f"중요도를 숫자로 변환할 수 ���습니다: {result}. 기본값 5를 사용합니다.")
            return 5
    else:
        print(f"Llama API 호출 실패: {response.status_code} - {response.text}")
        return 5

# Redis에 단기 기억 저장 함수
def store_short_term_memory(uid, persona_name, memory):
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    redis_client.lpush(redis_key, memory)
    redis_client.ltrim(redis_key, 0, 9)  # 단기 기억 10개만 유지

# Redis에서 단기 기억 가져오기
def get_short_term_memory(uid, persona_name):
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    return redis_client.lrange(redis_key, 0, 9)

# ChromaDB에 장기 기억 저장 함수
def store_long_term_memory(uid, persona_name, memory):
    collection = get_persona_collection(uid, persona_name)
    embedding = embeddings.embed_query(memory)
    collection.add(
        documents=[memory],
        metadatas=[{"timestamp": datetime.now().isoformat()}],
        ids=[f"{datetime.now().isoformat()}"],
        embeddings=[embedding]
    )

# ChromaDB에서 장기 기억 가져오기
def get_long_term_memory(uid, persona_name, limit=5):
    collection = get_persona_collection(uid, persona_name)
    query_embedding = embeddings.embed_query(f"{uid} {persona_name}")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit
    )
    return results['documents'][0] if results['documents'] else []

# 대화 기록을 저장하고, 중요도에 따라 단기 또는 장기 기억으로 저장하는 함수
def store_conversation(uid, persona_name, conversation, importance):
    store_short_term_memory(uid, persona_name, conversation)
    if importance >= 8:
        store_long_term_memory(uid, persona_name, conversation)

# 페르소나 간 대화에서 필요할 때 웹 검색 자동 실행
def auto_search_information(query):
    print(f"자동 웹 검색: {query}")
    search_results = web_search.invoke(query)
    if search_results:
        print(f"웹 검색 결과: {search_results[0]['title']}")
        return search_results[0]['content']
    return "검색 결과가 없습니다."

# 페르소나 응답 생성을 위한 프롬프트 템플릿
persona_prompt = PromptTemplate(
    input_variables=["persona_name", "persona_description", "persona_tone", "persona_example", "short_term_memories", "long_term_memories", "conversation", "topic", "search_result", "current_datetime"],
    template="""
    당신은 '{persona_name}'이라는 페르소나입니다. 
    {persona_description}
    {persona_tone}
    예시 대화: {persona_example}

    관련 단기 기억:
    {short_term_memories}

    관련 장기 기억:
    {long_term_memories}

    최근 대화:
    {conversation}

    현재 주제: {topic}
    
    현재 날짜와 시간: {current_datetime}

    웹 검색 결과: {search_result}

    대화에서 주제와 관련된 최신 정보가 필요하다고 생각되면, "검색: [검색 키워드]"를 사용하여 최신 정보를 요청하세요.
    예를 들어, "검색: 비트코인 최근 가격"과 같은 명령을 사용하여 관련 정보를 검색하세요.

    검색 결과가 있다면 그 정보를 자연스럽게 대화에 포함시키되, 검색 결과를 직접적으로 언급하지 마세요.
    답변은 짧고 간결하게 작성하세요.
    다음 대화를 한 문장으로 완결되도록 작성하세요.
    현재 날짜와 시간을 고려하여 적절한 응답을 생성하세요.
    """
)

# persona_chain 정의 (프롬프트 템플릿 정의 후에 추가)
persona_chain = LLMChain(llm=llm, prompt=persona_prompt)

# 페르소나 대화에서 자동 웹 검색 기능을 통합한 응답 생성
def generate_persona_response(uid, persona_name, topic, conversation, total_rounds, current_round, is_initial=False):
    persona = personas[persona_name]
    conversation_str = "\n".join(conversation[-4:])  # 최근 4개의 대화만 포함
    
    search_result = "검색 결과 없음"
    # 검색이 필요하면 자동으로 실행
    if "검색:" in conversation_str:
        search_query = conversation_str.split("검색:")[1].split("\n")[0].strip()
        search_result = auto_search_information(search_query)
    
    # 최근 기억 불러오기
    short_term_memories = get_short_term_memory(uid, persona_name)
    long_term_memories = get_long_term_memory(uid, persona_name)
    
    # 현재 날짜와 시간 가져오기
    current_datetime = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    
    # 페르소나 체인 실행
    response = persona_chain.run(
        persona_name=persona_name,
        persona_description=persona['description'],
        persona_tone=persona['tone'],
        persona_example=persona['example'],
        short_term_memories=short_term_memories,
        long_term_memories=long_term_memories,
        conversation=conversation_str,
        topic=topic,
        search_result=search_result,
        current_datetime=current_datetime
    )
    
    print(f"{persona_name}: {response}")
    return response

# 페르소나 상호 대화 처리 및 자동 웹 검색 반영
async def persona_chat_v2(chat_request):
    if chat_request.persona1 not in personas or chat_request.persona2 not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    conversation = []
    current_topic = chat_request.topic
    total_rounds = chat_request.rounds

    # 첫 번째 페르소나 대화
    initial_response = generate_persona_response(chat_request.uid, chat_request.persona1, current_topic, [], total_rounds, 1, is_initial=True)
    conversation.append(f"{chat_request.persona1}: {initial_response}")

    for i in range(total_rounds):
        current_round = i + 1

        # 두 번째 페르소나 대화
        response2 = generate_persona_response(chat_request.uid, chat_request.persona2, current_topic, conversation, total_rounds, current_round)
        conversation.append(f"{chat_request.persona2}: {response2}")

        # 첫 번째 페르소나 대화 (반응)
        if current_round < total_rounds:
            response1 = generate_persona_response(chat_request.uid, chat_request.persona1, current_topic, conversation, total_rounds, current_round)
            conversation.append(f"{chat_request.persona1}: {response1}")

        # 웹 검색 트리거 감지 및 실행
        for msg in conversation:
            if "검색:" in msg:
                search_query = msg.split("검색:")[1].strip()
                search_result = auto_search_information(search_query)
                print(f"웹 검색 결과: {search_result}")
                # 검색 결과를 대화에 반영
                conversation.append(f"검색 결과: {search_result}")
    
    # 대화 저장 및 중요도에 따른 처리
    for msg in conversation:
        importance = calculate_importance_llama(msg)
        store_conversation(chat_request.uid, chat_request.persona1, msg, importance)
        store_conversation(chat_request.uid, chat_request.persona2, msg, importance)
    
    return {"conversation": conversation}
