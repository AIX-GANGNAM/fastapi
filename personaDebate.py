from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_community.tools import TavilySearchResults
from datetime import datetime
from pydantic import BaseModel
import json
import asyncio
from typing import List, Dict
import pytz
from personas import personas
from service.personaChatVer3 import get_long_term_memory_tool, get_user_profile, get_user_events, calculate_importance_llama, summarize_content, store_short_term_memory, store_long_term_memory
from database import db
from firebase_admin import firestore
from service.smsservice import send_sms_service  # 상단에 import 추가

class StarEventRequest(BaseModel):
    uid: str
    eventId: str
    starred: bool
    time: str
    userPhone: str

class DebateMessage:
    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text
        self.timestamp = datetime.now(pytz.UTC).isoformat()
        self.isRead = True

class DebateRound:
    def __init__(self, topic: str, request: StarEventRequest):
        self.topic = topic
        self.request = request
        self.debate_history = []
        self.debate_ref = None
        self.initialize_debate()
        
    def initialize_debate(self):
        debate_ref = db.collection('personachat').document(self.request.uid).collection('debates').document()
        debate_ref.set({
            'title': self.topic,
            'eventId': self.request.eventId,
            'eventTime': self.request.time,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'status': 'in_progress',
            'finalSender': None,
            'finalMessage': None,
            'selectionReason': None
        })
        self.debate_ref = debate_ref
        
    def add_to_history(self, speaker: str, text: str, message_type: str = "message"):
        """토론 히스토리에 메시지 추가 및 메모리 저장"""
        message = DebateMessage(speaker, text)
        self.debate_history.append(message)
        
        # Firestore에 메시지 저장
        if self.debate_ref:
            self.debate_ref.collection('messages').add({
                'speaker': speaker,
                'text': text,
                'type': message_type,
                'timestamp': message.timestamp,
                'isRead': message.isRead
            })
        
        # 페르소나의 발언인 경우에만 메모리 저장
        if speaker != "Moderator":
            # 단기 기억에 저장
            store_short_term_memory(
                self.request.uid, 
                speaker, 
                f"{speaker}: {text}"
            )
            
            # 중요도 계산
            importance = calculate_importance_llama(text)
            
            # 중요도가 8 이상이면 장기 기억에 저장
            if importance >= 5:
                # 요약 생성
                summary = summarize_content(text)
                # 장기 기억에 저장
                store_long_term_memory(
                    self.request.uid,
                    speaker,
                    summary
                )
        
        print(f"\n{'🎭' if speaker == 'Moderator' else '💭'} {speaker}({personas[speaker]['realName']})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{text}")
        print(f"글자 수: {len(text)}자")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def print_sms(message_data: str) -> str:
    data = json.loads(message_data)
    message = data.get('message', '')
    sender = data.get('sender', '')
    
    if len(message) > 30:
        print("\n⚠️ 경고: 메시지가 30자를 초과하여 자동으로 수정됩니다.")
        message = message[:27] + "..."
    
    print(f"\n📱 최종 선정된 알림 메시지")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"│ 보내는 이: {sender}({personas[sender]['realName']})")
    print(f"│ 메시지: {message}")
    print(f"│ 글자 수: {len(message)}자")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    return json.dumps({
        "status": "success", 
        "message": message, 
        "sender": sender,
        "sender_real_name": personas[sender]['realName']
    })

async def generate_acceptance_speech(persona_name: str, event_request: StarEventRequest) -> str:
    persona_info = personas[persona_name]
    
    prompt = f"""당신은 {persona_name}({persona_info['realName']})입니다.

성격: {persona_info['description']}
말투: {persona_info['tone']}

방금 '{event_request.eventId}' 일정의 알림 메시지를 보내는 역할로 선정되었습니다.
다른 페르소나들에 대한 감사와 앞으로의 다짐을 당신의 성격과 말투로 표현해주세요.

요구사항:
- 감사의 마음을 표현
- 다른 페르소나들의 의견을 인정
- 메시지 전달에 대한 다짐
- 당신의 성격과 말투를 유지
- 100자 이내로 작성
"""
    
    response = await model.ainvoke(prompt)
    content = response.content
    
    if len(content) > 100:
        content = content[:97] + "..."
    
    return content

# Model and Tools 설정
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
web_search = TavilySearchResults(max_results=1)

tools = [
    Tool(
        name="Vote",
        func=lambda x: json.loads(x),
        description="페르소나들의 의견을 집계하고 투표하는 도구입니다. Input은 'votes' 배열(각 투표 내용)과 'reason'을 포함한 JSON 형식이어야 합니다."
    ),
    Tool(
        name="SendSMS",
        func=print_sms,
        description="최종 선정된 메시지를 출력합니다. Input은 'message'와 'sender'를 포함한 JSON 형식이어야 합니다."
    ),
    Tool(
        name="Search",
        func=web_search.invoke,
        description="날씨, 이벤트, 뉴스 등 현재 상황에 대한 정보를 검색할 때 사용합니다. 검색어에 'KST' 또는 '한국시간'을 포함해야 합니다."
    ),
    Tool(
        name="Current Time",
        func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="현재 날짜와 시간을 확인합니다. 모든 작업 전에 먼저 사용해야 합니다."
    ),
    Tool(
        name="Long Term Memory",
        func=get_long_term_memory_tool,
        description="페르소나의 장기 기억을 조회합니다. Input: {'uid': string, 'persona_name': string, 'query': string, 'limit': int}"
    ),
    Tool(
        name="Search Firestore for user profile",
        func=get_user_profile,
        description="사용자 프로필을 조회합니다. Input: {'uid': string}"
    ),
    Tool(
        name="owner's calendar",
        func=get_user_events,
        description="사용자의 일정을 조회합니다. Input: {'uid': string, 'date': string}"
    )
]
async def create_persona_response(name: str, event_request: StarEventRequest) -> str:
    persona_info = personas[name]
    
    # 시간 처리 수정
    try:
        # 이벤트 시간을 UTC로 변환
        event_time = datetime.fromisoformat(event_request.time.replace('Z', '+00:00'))
        if event_time.tzinfo is None:
            event_time = pytz.UTC.localize(event_time)
            
        # 현재 시간을 UTC로 가져오기
        current_time = datetime.now(pytz.UTC)
        
        # 시간 차이 계산
        time_diff = event_time - current_time
        
        # 남은 시간 계산
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        
        time_remaining = f"{days}일 " if days > 0 else ""
        time_remaining += f"{hours}시간 " if hours > 0 or days > 0 else ""
        time_remaining += f"{minutes}분" if minutes > 0 or (hours > 0 or days > 0) else "곧"
        
    except Exception as e:
        print(f"시간 계산 중 오류 발생: {str(e)}")
        time_remaining = "시간 계산 불가"
    
    prompt = f"""당신은 {name}({persona_info['realName']})입니다.

성격: {persona_info['description']}
말투: {persona_info['tone']}

현재 주인님의 일정 '{event_request.eventId}'에 대한 알림 메시지를 누가 보낼지 토론하고 있습니다.
일정 정보:
- 일정 제목: {event_request.eventId}
- 일정 시간: {event_request.time}
- 남은 시간: {time_remaining}

이벤트의 특성을 고려하여 당신이 이 메시지를 보내야 하는 이유를 주장해주세요.
주장 시 고려할 점:
1. 이벤트의 성격 (공식적/비공식적, 즐거운/진지한 등)
2. 시간의 중요성 (정시 도착 필요성, 여유 시간 등)
3. 준비 필요 사항
4. 이벤트에 적합한 감정과 태도

제약사항:
- 반드시 200자 이내로 의견을 제시해주세요
- 당신의 성격과 말투를 반영해주세요
- 다른 페르소나와의 차별점을 언급해주세요
"""
    
    response = await model.ainvoke(prompt)
    content = response.content
    
    if len(content) > 200:
        content = content[:197] + "..."
    
    return content

def send_final_message(request: StarEventRequest, result: Dict):
    """최종 선택된 메시지를 SMS로 전송"""
    try:
        sms_request = {
            "phone_number": request.userPhone.replace("-", ""),
            "message": result['message']
        }
        
        response = send_sms_service(json.dumps(sms_request))
        
        if response["status"] == "success":
            print("\n✉️ SMS 전송 완료")
            print(f"수신자: {request.userPhone}")
            print(f"메시지: {result['message']}")
            return True
        else:
            print(f"\n❌ SMS 전송 실패: {response['message']}")
            return False
            
    except Exception as e:
        print(f"\n❌ SMS 전송 중 오류 발생: {str(e)}")
        return False

async def run_persona_debate(event_request: StarEventRequest):
    # 시간대 처리 추가
    event_time = datetime.fromisoformat(event_request.time.replace('Z', '+00:00'))
    if event_time.tzinfo is None:
        event_time = pytz.UTC.localize(event_time)
    
    # 한국 시간으로 변환
    kst = pytz.timezone('Asia/Seoul')
    event_time_kst = event_time.astimezone(kst)
    formatted_time = event_time_kst.strftime("%Y년 %m월 %d일 %H시 %M분")
    
    print(f"\n🤖 페르소나 토론 시스템 시작")
    print(f"📅 일정: {event_request.eventId}")
    print(f"⏰ 시간: {formatted_time}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    debate = DebateRound(f"일정 알림 토론: {event_request.eventId}", event_request)
    
    # 이벤트 분석 및 토론 시작 메시지
    event_analysis = (
        f"[이벤트 분석]\n"
        f"일정: {event_request.eventId}\n"
        f"시간: {formatted_time}\n\n"
        f"이���트 특성 고려사항:\n"
        f"1. 이벤트의 성격과 중요도\n"
        f"2. 시간 관리의 중요성\n"
        f"3. 필요한 준비사항\n"
        f"4. 적절한 감정과 태도\n\n"
        f"각 페르소나는 이러한 특성을 고려하여 의견을 제시해주세요."
    )
    
    debate.add_to_history("Moderator", event_analysis, "analysis")
    
    # 각 페르소나의 의견 수집
    for name in personas.keys():
        response = await create_persona_response(name, event_request)
        debate.add_to_history(name, response, "opinion")
        await asyncio.sleep(1)
    
    # 토론 결과 정리 및 투표
    voting_message = (
        "모든 페르소나의 의견을 들었습니다.\n"
        "이제 각 의견을 종합하여 가장 적합한 페르소나를 선정하겠습니다."
    )
    debate.add_to_history("Moderator", voting_message, "voting")

    try:
        tool_names = [tool.name for tool in tools]
        
        executor = AgentExecutor(
            agent=create_react_agent(
                llm=model,
                tools=tools,
                prompt=PromptTemplate.from_template(debate_template)
            ),
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        # 토론 결과 실행
        result = await executor.ainvoke({
            "event_time": formatted_time,
            "event_id": event_request.eventId,
            "personas_details": "\n".join([
                f"{name}({info['realName']})\n- 성격: {info['description']}\n- 말투: {info['tone']}"
                for name, info in personas.items()
            ]),
            "debate_history": "\n".join([
                f"{msg.speaker}({personas[msg.speaker]['realName'] if msg.speaker != 'Moderator' else '진행자'}): {msg.text}"
                for msg in debate.debate_history
            ]),
            "tools": render_text_description(tools),
            "tool_names": ", ".join(tool_names),  # 도구 이름들을 문자열로 변환
            "agent_scratchpad": ""
        })

        # 결과 파싱
        # 결과 파싱
        output = result.get('output', '')
        try:
            final_data = await parse_final_answer(output)
            
            # 선정된 페르소나의 감사 인사 생성
            acceptance_speech = await generate_acceptance_speech(final_data['sender'], event_request)
            debate.add_to_history(final_data['sender'], acceptance_speech, "acceptance")

            # 최종 결과 발표
            final_announcement = (
                f"[토론 결과 발표]\n\n"
                f"✨ 선정된 페르소나: {final_data['sender']}({personas[final_data['sender']]['realName']})\n"
                f"📝 선정 이유: {final_data['reason']}\n"
                f"💌 최종 메시지: {final_data['message']}\n\n"
                f"토론에 참여해주신 모든 페르소나 여러분께 감사드립니다."
            )
            debate.add_to_history("Moderator", final_announcement, "final_result")

            # SMS 전송
            sms_data = {
                "message": final_data['message'],
                "sender": final_data['sender']
            }
            print_sms(json.dumps(sms_data))

            # Firestore 업데이트
            debate.debate_ref.update({
                'status': 'completed',
                'completedAt': firestore.SERVER_TIMESTAMP,
                'finalSender': final_data['sender'],
                'finalMessage': final_data['message'],
                'selectionReason': final_data['reason']
            })

            # SMS 전송
            if send_final_message(event_request, final_data):
                debate.add_to_history(
                    "Moderator", 
                    "SMS 전송이 완료되었습니다.", 
                    "system"
                )
            else:
                debate.add_to_history(
                    "Moderator", 
                    "SMS 전송에 실패했습니다.", 
                    "error"
                )

        except ValueError as e:
            print(f"결과 파싱 오류: {str(e)}")
            print("원본 출력:", output)
            raise

    except Exception as e:
        print(f"결과 처리 중 오류 발생: {str(e)}")
        debate.add_to_history(
            "Moderator",
            "토론 결과 처리 중 문제가 발생했습니다. 다시 시도해주세요.",
            "error"
        )
        raise

    return {
        "status": "success",
        "debate_id": debate.debate_ref.id,
        "debate_history": debate.debate_history,
        "final_result": final_data
    }


async def parse_final_answer(output: str) -> dict:
    try:
        # 구분자로 나누어진 최종 결정 부분 찾기
        if "======================================" in output:
            parts = output.split("======================================")
            if len(parts) >= 3:
                final_part = parts[1].strip()
            else:
                raise ValueError("최종 결정 구분자를 찾을 수 없습니다")
        else:
            raise ValueError("최종 결정 구분자를 찾을 수 없습니다")
        
        # 결과 데이터 초기화
        result = {
            'sender': None,
            'reason': None,
            'message': None
        }
        
        # 결과 파싱
        lines = final_part.split('\n')
        for line in lines:
            line = line.strip()
            if line:  # 빈 줄 무시
                if "선정된 페르소나:" in line:
                    result['sender'] = line.split("선정된 페르소나:")[1].strip().split('(')[0].strip()
                elif "선정 이유:" in line:
                    result['reason'] = line.split("선정 이유:")[1].strip()
                elif "최종 메시지:" in line:
                    result['message'] = line.split("최종 메시지:")[1].strip()
        
        # 결과 검증
        missing = [k for k, v in result.items() if not v]
        if missing:
            # 원본 출력 후 오류 발생
            print("\n⚠️ 파싱 오류 발생:")
            print("원본 텍스트:")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(output)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"누락된 항목: {', '.join(missing)}")
            raise ValueError(f"필수 항목이 누락되었습니다: {', '.join(missing)}")
        
        # 메시지 길이 검증
        if len(result['message']) > 30:
            result['message'] = result['message'][:27] + "..."
        
        print("\n🔍 파싱된 결과:")
        print(f"페르소나: {result['sender']}")
        print(f"선정 이유: {result['reason']}")
        print(f"메시지: {result['message']} ({len(result['message'])}자)")
        
        return result
        
    except Exception as e:
        print(f"\n⚠️ 파싱 시도한 원본 텍스트:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(output)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        raise ValueError(f"결과 파싱 중 오류 발생: {str(e)}")
    
# 토론 템플릿 수정
debate_template = '''당신은 5명의 페르소나가 토론하는 것을 진행하고 관리하는 토론 진행자입니다.

현재 상황:
- 일정 제목: {event_id}
- 일정 시간: {event_time}

[참여 페르소나]
{personas_details}

[지금까지의 토론 내용]
{debate_history}

당신의 역할:
1. 각 페르소나의 의견을 공정하게 평가
2. Vote 도구로 투표 진행
3. 투표 결과를 바탕으로 SendSMS 도구를 통해 메시지 전송
4. 반드시 최종 결정 양식에 맞춰 결과 발표

평가 기준:
1. 이벤트 성격과의 적합성
2. 메시지 전달의 효과성
3. 페르소나의 특성 활용도
4. 실용성과 명확성

사용 가능한 도구:
{tools}

다음 형식을 반드시 준수하세요:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: 반드시 아래 양식으로 최종 결정을 작성하세요:
========================================
선정된 페르소나: [이름]
선정 이유: [이유]
최종 메시지: [30자 이내 메시지]
========================================

{agent_scratchpad}'''

# 실행 예시
async def main():
    test_request = StarEventRequest(
        uid="DwgZh7Ud7STbVBnkyvK5kmxUIzw1",
        eventId="홍대 출발",
        starred=True,
        time="2024-10-27T22:00:00",
        userPhone="010-1234-5678"
    )
    
    try:
        result = await run_persona_debate(test_request)
        print("\n✨ 토론이 완료되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

