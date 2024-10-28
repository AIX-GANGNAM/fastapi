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
from service.services import send_expo_push_notification

model = ChatOpenAI(model="gpt-4o",temperature=0.5)
web_search = TavilySearchResults(max_results=1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

tools = [
    Tool(
        name="Search",
        func=web_search.invoke,
        description="useful for when you need to answer questions about current events. ALWAYS add 'KST' or '한국시간' when searching for event times or schedules."
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

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be a valid JSON string)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know what to say
Final Answer: your response in the following format:

Response: [your main response in Korean]
Context: [optional additional context]
Engagement: [optional follow-up question]

Remember to maintain consistent formatting and use proper JSON

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
    max_iterations=10,
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
    print("personaLoopChat > persona_chat_v2 > chat_request : ", chat_request)
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
            "tool_names": ", ".join([tool.name for tool in tools]),
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
        # 수정된 Response 패턴
        response_pattern = r'(?:Response|Context|Engagement): (.*?)(?=(?:Response|Context|Engagement):|Final Answer:|$)'
        responses = re.findall(response_pattern, output, re.DOTALL)
        
        # 응답이 없는 경우 기본 응답 저장
        if not responses:
            default_response = "죄송해요, 잠시 생각이 필요해요... 다시시도해주세요... 🤔"
            chat_ref.add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                'sender': persona_name,
                'message': default_response
            })
            await send_expo_push_notification(uid, persona_name, default_response, "Chat")
            print(f"persona_chat_v2 >Notification: {notification}")  
            return {"message": "Default response saved successfully"}
        
        # 응답 저장
        for response_text in responses:
            cleaned_response = response_text.strip()
            if cleaned_response:
                await asyncio.sleep(2)  # 딜레이 시간 단축
                chat_ref.add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    'sender': persona_name,
                    'message': cleaned_response
                })
                notification = await send_expo_push_notification(uid, persona_name, cleaned_response, "Chat")
                print(f"persona_chat_v2 > Notification: {notification}")
        return {"message": "Conversation completed successfully"}
        
    except Exception as e:
        print(f"Error during conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
