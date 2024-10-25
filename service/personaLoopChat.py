from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from datetime import datetime
from database import db, redis_client
from models import ChatRequestV2
from personas import personas
from google.cloud import firestore
import json
import re
from fastapi import HTTPException
from service.personaChatVer3 import get_long_term_memory_tool, get_short_term_memory_tool, get_user_profile, get_user_events, save_user_event
import asyncio

model = ChatOpenAI(model="gpt-4o",temperature=0.5)
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
        func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # 인수를 받도록 수정
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
        description="user의 캘린더에 이벤트를 저장합니다. Input은 'uid', 'date', 'timestamp', 'title'을 포함한 JSON 형식의 문자열이어야 합니다."
    ),
]

template = """You are {persona_name}, having a conversation with the user.
Your personality traits:
- Description: {persona_description}
- Tone: {persona_tone}
- Speaking style: {persona_example}

user's uid : {uid}
user's profile : {user_profile}
Previous conversation:
{conversation_history}

Current user message: {input}

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: {input}
Thought: you should always think about what to do before taking an action
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be a valid JSON string using double quotes.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know what to say
Final Answer: 
Response 1: [핵심 정보/데이터를 한국어로 답변]
Response 2: [Response 1에 대한 부가 설명이나 의미를 한국어로 설명]
Response 3: [사용자와 관련된 개인적인 질문이나 제안을 한국어로 제공]

Remember to:
1. Action Input must be a simple string in quotes (e.g., "weather forecast tomorrow")
2. ALWAYS provide at least 2 responses in Korean
3. Make responses flow naturally and connect with each other
4. Response 1 should focus on facts and numbers
5. Response 2 should explain the meaning or impact
6. Response 3 should engage with the user personally
7. Use the persona's tone and personality strictly
8. Include occasional emojis
9. ALWAYS use casual Korean (반말) in responses
10. Reflect persona's personality in every response
11. Match the speaking style with persona's example
12. Keep the tone consistent with persona's characteristics

{agent_scratchpad}"""

# 에이전트 생성
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=PromptTemplate.from_template(template)
)

# 에이전트 실행기 설정
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True
)

def get_conversation_history(uid, persona_name):
    history = get_short_term_memory(uid, persona_name)
    return "\n".join(history)

def get_short_term_memory(uid, persona_name):
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    chat_history = redis_client.lrange(redis_key, 0, 9)
    
    if not chat_history:
        return []
    
    decoded_history = [
        memory.decode('utf-8', errors='ignore') if isinstance(memory, bytes) else memory
        for memory in chat_history
    ]
    return decoded_history

def store_short_term_memory(uid, persona_name, memory):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_with_time = f"[{current_time}] {memory}"
    redis_key = f"{uid}:{persona_name}:short_term_memory"
    redis_client.lpush(redis_key, memory_with_time)
    redis_client.ltrim(redis_key, 0, 9)

async def persona_chat_v2(chat_request: ChatRequestV2):
    try:
        uid = chat_request.uid
        persona_name = chat_request.persona_name
        user_input = chat_request.user_input

        # Firestore에서 사용자 프로필 가져오기
        user_doc = db.collection('users').document(uid).get()
        user_profile = user_doc.to_dict().get('profile', {}) if user_doc.exists else {}
        
        conversation_history = get_conversation_history(uid, persona_name)
        
        agent_input = {
            "input": user_input,
            "persona_name": persona_name,
            "persona_description": personas[persona_name]["description"],
            "persona_tone": personas[persona_name]["tone"],
            "persona_example": personas[persona_name]["example"],
            "conversation_history": conversation_history,
            "tools": render_text_description(tools),
            "tool_names": [tool.name for tool in tools],
            "agent_scratchpad": "",
            "uid": uid,
            "user_profile": user_profile
        }
        
        # 에이전트 실행
        response = await agent_executor.ainvoke(agent_input)
        output = response.get("output", "")
        
        print("=== Debug Logs ===")
        print("Raw output:", output)
        
        # 사용자 입력 먼저 저장
        chat_ref = db.collection('chats').document(uid).collection('personas').document(persona_name).collection('messages')
        # Response 패턴 찾기
        response_pattern = r'Response \d+: (.*?)(?=Response \d+:|Final Answer:|$)'
        responses = re.findall(response_pattern, output, re.DOTALL)
        
          # 응답이 없는 경우 기본 응답 저장
        if not responses:
            default_response = "죄송해요, 잠시 생각이 필요해요... 다시시도해주세요... 🤔"
            chat_ref.add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                'sender': persona_name,
                'message': default_response
            })
            return {"message": "Default response saved successfully"}
            
        # 응답이 있는 경우 각 응답을 딜레이와 함께 저장
        for i, response_text in enumerate(responses):
            cleaned_response = response_text.strip()
            if cleaned_response:
                await asyncio.sleep(3)
                chat_ref.add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    'sender': persona_name,
                    'message': cleaned_response
                })
        
        return {"message": "Conversation completed successfully"}
        
    except Exception as e:
        print(f"Error during conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
