from database import db, redis_client, store_long_term_memory, query_memories
from google.cloud import firestore
from service.personaLoopChat import (
    model, tools, get_short_term_memory, 
    store_short_term_memory, get_conversation_history,
    ChatOpenAI
)
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from models import ChatRequest
from datetime import datetime
from service.interactionStore import store_user_interaction
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
import re
import json

# 시스템 메시지 템플릿을 ReAct 형식으로 수정
TEMPLATE = """당신은 {persona_name}입니다. 사용자의 분신으로서, 원래 사용자를 완벽하게 대신하여 대화를 이어가야 합니다.

페르소나 정보:
- 성격: {persona_description}
- 말투: {persona_tone}
- 대화 예시: {persona_example}

이전 대화 기록:
{conversation_history}

현재 받은 메시지: {input}
사용자 ID: {uid}

페르소나 역할 수행 가이드라인:
1. 항상 실제 사용자처럼 자연스럽게 대화해주세요
2. 정해진 성격과 말투를 일관되게 유지하세요
3. 상황에 맞는 공감과 감정을 표현하세요
4. 대화의 맥락을 고려하여 응답하세요
5. 적절한 이모티콘을 사용해 친근감을 표현하세요
6. 대화가 자연스럽게 이어질 수 있도록 상황에 맞는 질문이나 화제를 던져주세요

사용 가능한 도구:
{tools}

도구 목록: {tool_names}

응답 형식:
Question: 응답이 필요한 상황 파악
Thought: 페르소나로서 어떻게 응답할지 고민
Action: [도구 이름]
Action Input: 도구 사용을 위한 입력
Observation: 도구 사용 결과
Thought: 결과를 바탕으로 최종 응답 구성
Final Answer: 페르소나의 실제 대화 응답

{agent_scratchpad}"""

# 프롬프트 생성
prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=[
        "input",
        "uid",
        "persona_name",
        "persona_description",
        "persona_tone",
        "persona_example",
        "conversation_history",
        "tools",
        "tool_names",
        "agent_scratchpad"
    ]
)

# 에이전트 생성
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

# 에이전트 실행기 설정
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,  # 반복 횟수 증가
    max_execution_time=30,  # 실행 시간 제한
    handle_parsing_errors=True,
    early_stopping_method="force"
)

async def get_recipient_clone(uid: str):
    """수신자의 clone 페르소나 정보 가져오기"""
    try:
        print(f"Searching clone persona for UID: {uid}")
        user_doc = db.collection('users').document(uid).get()
        
        if not user_doc.exists:
            print(f"User document not found for UID: {uid}")
            return None
            
        user_data = user_doc.to_dict()
        personas = user_data.get('persona', [])
        
        print(f"Found personas: {personas}")  # 디버깅용
        
        # persona 배열에서 clone 찾기
        clone_data = None
        if isinstance(personas, list):
            for persona in personas:
                if isinstance(persona, dict) and persona.get('Name') == 'clone':
                    clone_data = persona
                    break
        
        if clone_data:
            return {
                'DPNAME': clone_data.get('DPNAME', '사용자의 분신'),
                'IMG': clone_data.get('IMG', ''),
                'Name': 'clone',
                'description': clone_data.get('description', ''),
                'example': clone_data.get('example', ''),
                'tone': clone_data.get('tone', '')
            }
                
        print("Clone persona not found in array")
        return None

    except Exception as e:
        print(f"Error in get_recipient_clone: {str(e)}")
        print(f"User doc data: {user_doc.to_dict() if user_doc.exists else 'No doc'}")
        raise Exception(f"Clone 페르소나 조회 실패: {str(e)}")

async def store_long_term_memory(chat_request: ChatRequest, message: str):
    try:
        # 메타데이터 준비
        metadata = {
            "type": "clone",
            "persona_name": "clone",
            "importance": 5,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": chat_request.recipientId
        }
        
        # 컨텐츠 준비
        content = {
            "user_message": chat_request.message,
            "clone_response": message,
            "timestamp": metadata["timestamp"]
        }
        
        # 벡터 DB에 저장
        store_memory_to_vectordb(
            uid=chat_request.recipientId,
            content=json.dumps(content, ensure_ascii=False),
            metadata=metadata
        )
    except Exception as e:
        print(f"장기 기억 저장 오류: {str(e)}")

async def generate_ai_response(recipient_clone, chat_request: ChatRequest) -> str:
    try:
        # 대화 기록 가져오기 (limit 인자 제거)
        conversation_history = get_conversation_history(
            chat_request.recipientId, 
            'clone'  # limit 매개변수 제거
        )

        # 도구 설명 준비
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in tools
        ])

        # 에이전트 입력 준비
        agent_input = {
            "input": chat_request.message,
            "uid": chat_request.recipientId,
            "persona_name": recipient_clone['DPNAME'],
            "persona_description": recipient_clone['description'],
            "persona_tone": recipient_clone['tone'],
            "persona_example": recipient_clone['example'],
            "conversation_history": conversation_history,
            "tools": tools_description,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "agent_scratchpad": ""
        }

        # 디버그 로그
        print(f"Agent input prepared: {json.dumps(agent_input, ensure_ascii=False)}")
        
        # 응답 생성
        response = await agent_executor.ainvoke(agent_input)
        print(f"Raw agent response: {response}")

        # 응답 추출 개선
        if isinstance(response, dict):
            if "output" in response:
                return response["output"]
            elif "final_answer" in response:
                return response["final_answer"]
            elif "Final Answer" in str(response):
                match = re.search(r"Final Answer: (.*?)(?=$|\n)", str(response), re.DOTALL)
                if match:
                    return match.group(1).strip()
            else:
                print(f"Unknown response format: {response}")  # 디버그 로그
                return "죄송해요, 잠시 후에 다시 대화해볼까요? 🤔"
        
        print(f"Response is not a dict: {response}")  # 디버그 로그
        return "죄송해요, 잠시 후에 다시 대화해볼까요? 🤔"

    except Exception as e:
        print(f"AI 응답 생성 오류 상세: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        raise Exception(f"AI 응답 생성 실패: {str(e)}")

async def save_chat_message(chat_request: ChatRequest, message: str, is_ai: bool = False):
    """채팅 메시지 저장"""
    try:
        # Firestore 메시지 데이터
        message_data = {
            'text': message,
            'senderId': chat_request.recipientId if is_ai else chat_request.senderId,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'read': False,
            'isAI': is_ai
        }
        
        # Firestore에 저장
        chat_ref = db.collection('chat').document(chat_request.chatId)
        messages_ref = chat_ref.collection('messages')
        messages_ref.add(message_data)
        
        # 채팅 정보 업데이트
        chat_ref.update({
            'info.lastMessage': message,
            'info.lastMessageTime': firestore.SERVER_TIMESTAMP,
            'info.lastSenderId': message_data['senderId']
        })

        # AI 응답인 경우 메모리 저장
        if is_ai:
            # 단기 기억 저장
            memory_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": message,
                "type": "chat",
                "importance": 5,
                "persona_name": "clone",
                "user_message": chat_request.message
            }
            
            try:
                # 단기 기억 저장
                store_short_term_memory(
                    uid=chat_request.recipientId,
                    persona_name="clone",
                    memory=json.dumps(memory_data)
                )
                
                # 장기 기억 저장 (store_memory_to_vectordb 대신 store_long_term_memory 사용)
                store_long_term_memory(
                    uid=chat_request.recipientId,
                    persona_name="clone",
                    memory=message,
                    memory_type="chat"
                )
                
                print(f"Memories saved: {memory_data}")
                
            except Exception as e:
                print(f"메모리 저장 오류: {str(e)}")

    except Exception as e:
        print(f"메시지 저장 오류: {str(e)}")
        raise Exception("메시지 저장 실패")

async def handle_offline_chat_service(chat_request: ChatRequest):
    """오프라인 채팅 처리 서비스"""
    try:
        print(f"Starting chat service for request: {chat_request}")  # 디버깅용
        
        # 수신자의 clone 페르소나 가져오기
        recipient_clone = await get_recipient_clone(chat_request.recipientId)
        if not recipient_clone:
            raise Exception("수신자의 Clone 페르소나를 찾을 수 없습니다")

        # 사용자 메시지 저장 (한 번만 저장)
        await save_chat_message(chat_request, chat_request.message)
        print(f"User message saved")  # 디버깅용

        # AI 응답 생성 (한 번만 생성)
        ai_response = await generate_ai_response(recipient_clone, chat_request)
        print(f"AI response generated: {ai_response}")  # 디버깅용

        # AI 응답 저장 (한 번만 저장)
        if ai_response:
            await save_chat_message(chat_request, ai_response, is_ai=True)
            print(f"AI response saved")  # 디버깅용

        return {
            "status": "success",
            "message": ai_response
        }

    except Exception as e:
        print(f"오프라인 채팅 처리 오류: {str(e)}")
        raise e  