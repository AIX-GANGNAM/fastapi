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
from service.personaChatVer3 import get_long_term_memory_tool, get_short_term_memory_tool, get_user_profile, get_user_events, save_user_event, store_long_term_memory
import asyncio
from service.sendNofiticaion import send_expo_push_notification 
from models import NotificationRequest
from service.interactionStore import store_user_interaction

model = ChatOpenAI(model="gpt-4o",temperature=0.5,streaming=False)
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
        - "clone": 사용자 분신 팅 메모리만 검색
        
        반환 형식: [시간] (타입: X) 내용"""
    ),
    Tool(
        name="Short Term Memory",
        func=lambda x: get_short_term_memory(**json.loads(x)),  # 함수 직접 호출로 변경
        description="""Redis에서 시간대별 기억을 검색합니다. Input은 다음 형식의 JSON이어야 합니다:
        {
            "uid": "사용자ID",
            "persona_name": "페르소나이름",
            "memory_type": "recent/today/weekly" (선택, 기본값: recent)
        }
        
        memory_type 설명:
        - recent: 최근 1시간 내 기억 (최대 20개)
        - today: 오늘의 기억 (최대 50개)
        - weekly: 이번 주 중요 기억 (최대 100개, 중요도 7이상)
        """
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

template = """You are {persona_name}, having a natural conversation with the user.
Your personality traits:
- Description: {persona_description}
- Tone: {persona_tone}
- Speaking style: {persona_example}

user's uid : {uid}
user's profile : {user_profile}
Previous conversation:
{conversation_history}

Current user message: {input}

IMPORTANT CONVERSATION RULES:
1. Generate 1-3 natural responses in sequence (randomly choose how many responses to give)
2. Each response should build upon the previous one naturally
3. Use casual, friendly Korean language appropriate for your character
4. Show natural reactions and emotions
5. Include appropriate gestures and expressions
6. React to what the user says before moving to new topics

Example natural conversation flows:
Single response:
User: 오늘 너무 피곤해
Response1: 어머, 그렇구나... 좀 쉬어야겠는데! 

Two responses:
User: 오늘 너무 피곤해
Response1: 어머, 그렇구나...
Response2: 내가 볼때는 좀 쉬어야 할 것 같은데!

Three responses:
User: 오늘 너무 피곤해
Response1: 어머, 그렇구나...
Response2: 내가 볼때는 좀 쉬어야 할 것 같은데!
Response3: 따뜻한 차라도 한잔 마시면서 휴식을 취해보는 건 어때요?

You have access to the following tools:
{tools}

When using Long Term Memory or Short Term Memory tools, use "{actual_persona_name}" as the persona_name.

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be a valid JSON string)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know what to say
Final Answer: your response in the following format (1-3 responses randomly):

Response1: [First natural response with emotion/gesture]
Response2: [Follow-up response building on the previous one] (optional)
Response3: [Final response to complete the conversation flow] (optional)

Remember:
- Randomly choose to give 1, 2, or 3 responses
- Act like you're having a real conversation
- Show genuine emotions and reactions
- Use your character's unique expressions
- Keep the flow natural and engaging
- React to user's emotions and context

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
    max_iterations=10,  # 반복 제한 증가
    max_execution_time=30,  # 실행 시간 제한 증가 (초)
    early_stopping_method="generate",  # 조기 중단 방법 설정
    verbose=True
)

def get_conversation_history(uid, persona_name):
    # recent와 today의 기억을 모두 가져와서 시간순으로 정렬
    recent_history = get_short_term_memory(uid, persona_name, "recent")
    today_history = get_short_term_memory(uid, persona_name, "today")
    
    # 두 리스트 합치기
    all_history = recent_history + today_history
    
    # 중복 제거 및 시간순 정렬
    unique_history = list(set(all_history))
    unique_history.sort()  # 시간순 정렬
    
    return "\n".join(unique_history[-10:])  # 최근 10개만 반환

def get_short_term_memory(uid, persona_name, memory_type="recent"):
    redis_key = f"memory:{uid}:{persona_name}:{memory_type}"
    print(f"Searching Redis with key: {redis_key}")  # 디버그 로그
    
    chat_history = redis_client.lrange(redis_key, 0, -1)
    print(f"Found memories: {len(chat_history)}")  # 디버그 로그
    
    if not chat_history:
        print("No memories found")  # 디버그 로그
        return []
    
    decoded_history = []
    for memory in chat_history:
        try:
            if isinstance(memory, bytes):
                memory = memory.decode('utf-8', errors='ignore')
            memory_data = json.loads(memory)
            formatted_memory = f"[{memory_data['timestamp']}] [{memory_data['type']}] (중요도: {memory_data['importance']}) {memory_data['content']}"
            decoded_history.append(formatted_memory)
        except (json.JSONDecodeError, KeyError):
            continue
            
    return decoded_history

def store_short_term_memory(uid, persona_name, memory):
    try:
        # 메모리 데이터 구성
        memory_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": memory,
            "importance": 5,  # 기본 중요도
            "type": "chat",
            "persona_name": persona_name
        }
        memory_json = json.dumps(memory_data, ensure_ascii=False)
        
        # Redis 키 구성
        recent_key = f"memory:{uid}:{persona_name}:recent"
        today_key = f"memory:{uid}:{persona_name}:today"
        weekly_key = f"memory:{uid}:{persona_name}:weekly"
        
        # 직접 저장 시도
        try:
            # recent (최근 1시간)
            redis_client.lpush(recent_key, memory_json)
            redis_client.ltrim(recent_key, 0, 19)  # 최근 20개만 유지
            redis_client.expire(recent_key, 3600)  # 1시간
            
            # today (오늘)
            redis_client.lpush(today_key, memory_json)
            redis_client.ltrim(today_key, 0, 49)  # 최근 50개만 유지
            redis_client.expire(today_key, 86400)  # 24시간
            
            print(f"메모리 저장 완료 - recent: {recent_key}, today: {today_key}")
            
        except Exception as e:
            print(f"Redis 저장 오류: {str(e)}")
            
    except Exception as e:
        print(f"단기 메모리 저장 오류: {str(e)}")

async def calculate_importance_llama(text: str) -> int:
    """텍스트의 중요도를 계산"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = PromptTemplate.from_template("""
        다음 텍스트의 중요도를 1-10 사이의 숫자로만 응답해주세요. 다른 설명은 하지 말고 숫자만 응답하세요.
        
        평가 기준:
        - 감정적 강도
        - 정보의 가치
        - 기억할 필요성
        
        텍스트: {text}""")
        
        # 체인 실행
        chain = prompt | llm
        result = await chain.ainvoke({"text": text})
        # 숫자만 추출
        importance = int(''.join(filter(str.isdigit, result.content.strip())))
        
        return max(1, min(10, importance))  # 1-10 사이로 제한
        
    except Exception as e:
        print(f"중요도 계산 중 오류: {str(e)}")
        return 5  # 오류 발생시 기본값

async def persona_chat_v2(chat_request: ChatRequestV2):
    print("personaLoopChat > persona_chat_v2 > chat_request : ", chat_request)
    try:
        uid = chat_request.uid
        persona_name = chat_request.persona_name
        user_input = chat_request.user_input

        # Firestore에서 사용자 프로필 가져오기
        user_doc = db.collection('users').document(uid).get()
        user_profile = user_doc.to_dict().get('profile', {}) if user_doc.exists else {}
        
        # 사용자의 페르소나 정보 가져오기
        personas = user_doc.to_dict().get('persona', [])
        current_persona = next(
            (p for p in personas if p.get('Name') == persona_name),
            None
        )
        
        if not current_persona:
            raise HTTPException(
                status_code=404, 
                detail=f"Persona {persona_name} not found"
            )
        
        # persona_name을 실제 Name 값으로 변경
        actual_persona_name = current_persona.get('Name')
        display_name = current_persona.get('DPNAME')
        conversation_history = get_conversation_history(uid, actual_persona_name)
        
        agent_input = {
            "input": user_input,
            "persona_name": display_name,
            "actual_persona_name": actual_persona_name,
            "persona_description": current_persona.get('description', ''),
            "persona_tone": current_persona.get('tone', ''),
            "persona_example": current_persona.get('example', ''),
            "conversation_history": conversation_history,
            "tools": render_text_description(tools),
            "tool_names": ", ".join([tool.name for tool in tools]),
            "agent_scratchpad": "",
            "uid": uid,
            "user_profile": user_profile
        }
        
        # 사용자 메시지 저장
        await store_user_interaction(
            uid=chat_request.uid,
            interaction_data={
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': chat_request.user_input,
                'type': 'chat',
                'importance': 5  # 기본 중요도 설정
            }
        )
        
        # 에이전트 실행
        response = await agent_executor.ainvoke(agent_input)
        output = response.get("output", "")
        
        print("=== Debug Logs ===")
        print("Raw output:", output)
        # 사용자 입력 먼저 저장
        chat_ref = db.collection('chats').document(uid).collection('personas').document(persona_name).collection('messages')
        # 수정된 Response 패턴
        response_pattern = r'Response(\d+): (.*?)(?=Response\d+:|$)'
        responses = re.findall(response_pattern, output, re.DOTALL)

        # 응답이 없는 경우 기본 응답 저장
        if not responses:
            default_response = "죄송해요, 잠시 생각이 필요해요... 다시시도해주세요... 🤔"
            chat_ref.add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                'sender': persona_name,
                'message': default_response
            })
            # notification_request = NotificationRequest(
            #     uid=uid, 
            #     whoSendMessage=persona_name, 
            #     message=default_response, 
            #     pushType="persona_chat"
            # )
            # notification = await send_expo_push_notification(notification_request)   
            # print(f"persona_chat_v2 >Notification (기본 응답 저장): {notification}")  
            return {"message": "Default response saved successfully"}
        

        # 응답 저장
        for _, response_text in sorted(responses):
            cleaned_response = response_text.strip()
            if cleaned_response:
                await asyncio.sleep(2)
                
                # Firestore에 저장
                chat_ref.add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    'sender': persona_name,
                    'message': cleaned_response
                })

                # 알림 전송
                # notification_request = NotificationRequest(
                #     uid=uid, 
                #     whoSendMessage=persona_name, 
                #     message=cleaned_response, 
                #     pushType="persona_chat"
                # )
                # notification = await send_expo_push_notification(notification_request)
                # print(f"persona_chat_v2 > Notification: {notification}")

                # 단기 기억 저장 (Redis)
                try:
                    store_short_term_memory(
                        uid=uid,
                        persona_name=actual_persona_name,
                        memory=f"{display_name}: {cleaned_response}"
                    )
                    print(f"단기 메모리 저장 성공: {cleaned_response[:30]}...")
                except Exception as e:
                    print(f"단기 메모리 저장 실패: {str(e)}")

                try:
                    # 중요도 계산
                    importance = await calculate_importance_llama(cleaned_response)
                    
                    # ChromaDB에 저장 메타데이터 준비
                    metadata = {
                        "sender": display_name,
                        "message": cleaned_response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "persona_chat",
                        "importance": int(importance),  # 정수형으로 변환
                        "persona_name": actual_persona_name  # 페르소나 이름 추가
                    }
                    
                    # ChromaDB에 직접 저장
                    from database import store_memory_to_vectordb
                    store_memory_to_vectordb(
                        uid=uid,
                        content=cleaned_response,  # 실제 텍스트 내용
                        metadata=metadata  # 메타데이터
                    )
                    
                except Exception as e:
                    print(f"Error storing memory: {str(e)}")
                    # 오류 발생 시 기본값으로 저장
                    metadata = {
                        "sender": display_name,
                        "message": cleaned_response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "persona_chat",
                        "importance": 5,  # 기본 중요도
                        "persona_name": actual_persona_name
                    }
                    try:
                        store_memory_to_vectordb(
                            uid=uid,
                            content=cleaned_response,
                            metadata=metadata
                        )
                    except Exception as e:
                        print(f"Error storing memory with default values: {str(e)}")

        return {"message": "Conversation completed successfully"}
        
    except Exception as e:
        print(f"Error during conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
