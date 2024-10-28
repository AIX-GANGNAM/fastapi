from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_community.tools import TavilySearchResults
from datetime import datetime
import json
import asyncio
from typing import Optional, List
import pytz
from pydantic import BaseModel
import re

from personas import personas
from service.personaLoopChat import model
from database import db
from firebase_admin import firestore

class DebateMessage:
    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text
        self.timestamp = datetime.now(pytz.UTC).isoformat()
        self.isRead = True

class FeedCommentRequest(BaseModel):
    uid: str                    # 게시물 작성자 ID
    feed_id: str                # 게시물 ID
    image_description: str      # 이미지 설명
    caption: str                # 게시물 내용
    comment_count: int = 2      # 선정할 댓글 작성자 수

def print_vote_result(vote_data: str) -> str:
    """투표 결과 출력 및 처리"""
    try:
        data = json.loads(vote_data)
        result = {
            "votes": data.get('votes', []),
            "reason": data.get('reason', ''),
            "selected_personas": []
        }
        
        # 0.7점 이상인 페르소나 선정 (최대 3명)
        selected = sorted(
            [v for v in data['votes'] if v.get('score', 0) >= 0.7],
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:3]
        
        result["selected_personas"] = [s['persona'] for s in selected]
        
        print("\n🗳️ 투표 결과:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for vote in data['votes']:
            print(f"- {vote['persona']}: {vote.get('score', 0):.2f}점")
        print(f"\n선정된 페르소나:")
        for persona in result["selected_personas"]:
            print(f"- {persona}")
        print(f"\n선정 이유: {data.get('reason', '')}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return json.dumps(result)
    except Exception as e:
        print(f"투표 결과 처리 중 오류 발생: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

# 도구 정의
# 투표 도구 수정
tools = [
    Tool(
        name="Vote",
        func=print_vote_result,
        description="""페르소나들의 점수를 평가하고 투표하는 도구입니다.
        Input format: {
            "votes": [
                {"persona": "이름", "score": 0.0~1.0}, 
                ...
            ],
            "reason": "선정 이유"
        }"""
    ),
    Tool(
        name="Current Time",
        func=lambda _: datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
        description="현재 시간을 확인합니다."
    )
]

class CommentDebateRound:
    def __init__(self, request: FeedCommentRequest):
        self.request = request
        self.debate_history = []
        self.debate_ref = None
        self.initialize_debate()
        
    def initialize_debate(self):
        debate_ref = db.collection('personachat').document(self.request.uid)\
            .collection('debates').document()
        
        topic = f"피드 '{self.request.caption[:20]}...'에 대한 댓글 토론"
        
        debate_ref.set({
            'title': topic,
            'feed_id': self.request.feed_id,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'status': 'in_progress',
            'finalSender': None,
            'finalMessage': None,
            'selectionReason': None,
            'selected_personas': []
        })
        self.debate_ref = debate_ref

    def add_to_history(self, speaker: str, text: str, message_type: str = "opinion"):
        if len(text) > 200:
            text = text[:197] + "..."
            
        current_time = firestore.SERVER_TIMESTAMP
        speaker_name = "진행자" if speaker == "Moderator" else personas[speaker]['realName']
        
        self.debate_ref.collection('messages').add({
            'speaker': speaker,
            'speakerName': speaker_name,
            'text': text,
            'messageType': message_type,
            'timestamp': current_time,
            'isRead': True,
            'charCount': len(text)
        })
        
        message = DebateMessage(speaker, text)
        self.debate_history.append(message)
        
        print(f"\n{'🎭' if speaker == 'Moderator' else '💭'} {speaker}({speaker_name})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{text}")
        print(f"글자 수: {len(text)}자")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def update_debate_result(self, selected_personas: List[str], selection_reasons: dict):
        self.debate_ref.update({
            'status': 'completed',
            'selected_personas': selected_personas,
            'selection_reasons': selection_reasons,
            'completedAt': firestore.SERVER_TIMESTAMP
        })

async def create_persona_feed_response(name: str, request: FeedCommentRequest) -> str:
    """각 페르소나의 의견 생성"""
    persona_info = personas[name]
    
    prompt = f"""당신은 {name}({persona_info['realName']})입니다.

성격: {persona_info['description']}
말투: {persona_info['tone']}

다음 게시물에 대해 댓글을 달고 싶은 이유를 설명해주세요:

이미지 설명: {request.image_description}
글 내용: {request.caption}

주장 시 고려할 점:
1. 게시물의 성격 (감성적/정보적/일상적 등)
2. 이미지와 글의 분위기
3. 어떤 종류의 댓글이 적절할지
4. 당신의 성격이 이 게시물과 얼마나 잘 어울리는지

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

async def generate_comment(persona_name: str, request: FeedCommentRequest, direction: str) -> str:
    """댓글 생성 함수"""
    persona_info = personas[persona_name]
    
    prompt = f"""당신은 {persona_name}({persona_info['realName']})입니다.

성격: {persona_info['description']}
말투: {persona_info['tone']}

다음 게시물에 대한 댓글을 작성해주세요:
이미지 설명: {request.image_description}
글 내용: {request.caption}

댓글 작성 방향: {direction}

요구사항:
1. 당신의 성격과 말투를 반영해주세요
2. 게시물의 분위기와 내용에 어울리게 작성해주세요
3. 100자 이내로 작성해주세요
4. 공감과 관심을 표현해주세요
"""
    
    response = await model.ainvoke(prompt)
    comment = response.content
        
    return comment

async def save_debate_result(debate_ref, final_data: dict, comments: list):
    """토론 결과 저장"""
    try:
        debate_ref.update({
            'status': 'completed',
            'completedAt': firestore.SERVER_TIMESTAMP,
            'selected_personas': final_data['selected_personas'],
            'scores': final_data['scores'],
            'comments': comments
        })
    except Exception as e:
        print(f"토론 결과 저장 중 오류 발생: {str(e)}")
        raise

# 토론 진행자 프롬프트 수정
moderator_template = """당신은 Instagram 피드 댓글을 평가하는 토론 진행자입니다.

현재 게시물:
- 이미지: {image_description}
- 내용: {caption}

토론 기록:
{debate_history}

각 페르소나의 의견을 평가하여 다음을 수행하세요:
1. 각 페르소나에게 0.0~1.0 사이의 점수 부여
2. 0.7점 이상인 페르소나 중 최대 3명 선정
3. 선정된 페르소나별로 댓글 작성 방향 제시

평가 기준:
- 게시물 성격과의 적합성 (40%)
- 예상되는 댓글의 질과 적절성 (30%)
- 다른 페르소나와의 차별성 (30%)

사용 가능한 도구:
{tools}

다음 형식을 엄격히 준수하세요:

Question: the task you need to complete
Thought: think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: must be valid JSON string containing votes array with persona, score, and reason
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know what to say
Final Answer: your response in the following format:

Response: [평가 결과 요약 (한국어)]
Selected Personas: [선정된 페르소나 목록]
Scores: {{"페르소나1": 점수1, "페르소나2": 점수2, ...}}
Directions: {{"페르소나1": "댓글 작성 방향1", "페르소나2": "댓글 작성 방향2", ...}}

{agent_scratchpad}"""

# 에이전트 생성
moderator_agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=PromptTemplate.from_template(moderator_template)
)

# 에이전트 실행기 설정
agent_executor = AgentExecutor(
    agent=moderator_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True
)

async def parse_comment_debate_result(output: str) -> dict:
    """토론 결과 파싱"""
    try:
        # Response 패턴 매칭
        response_pattern = r'Response: (.*?)(?=Selected Personas:|$)'
        selected_pattern = r'Selected Personas: \[(.*?)\]'  # 대괄호 안의 내용만 캡처
        scores_pattern = r'Scores: ({.*?})'  # 중괄호 전체를 캡처
        directions_pattern = r'Directions: ({.*?})$'  # 중괄호 전체를 캡처

        response = re.search(response_pattern, output, re.DOTALL)
        selected = re.search(selected_pattern, output, re.DOTALL)
        scores = re.search(scores_pattern, output, re.DOTALL)
        directions = re.search(directions_pattern, output, re.DOTALL)

        # selected_personas 처리 개선
        selected_personas = []
        if selected and selected.group(1):
            # 쉼표로 분리하고 따옴표와 공백 제거
            selected_personas = [
                p.strip().strip('"').strip("'") 
                for p in selected.group(1).split(',')
            ]

        # scores와 directions는 json.loads 전에 문자열 정리
        scores_dict = {}
        directions_dict = {}
        
        if scores and scores.group(1):
            try:
                scores_dict = json.loads(scores.group(1))
            except json.JSONDecodeError:
                print("Scores 파싱 실패")
                
        if directions and directions.group(1):
            try:
                directions_dict = json.loads(directions.group(1))
            except json.JSONDecodeError:
                print("Directions 파싱 실패")

        return {
            'response': response.group(1).strip() if response else '',
            'selected_personas': selected_personas,
            'scores': scores_dict,
            'comment_directions': directions_dict
        }
    except Exception as e:
        print(f"결과 파싱 중 오류 발생: {str(e)}")
        # 기본값 반환
        return {
            'response': '',
            'selected_personas': [],
            'scores': {},
            'comment_directions': {}
        }

async def save_comment_to_db(persona: str, comment: str, feed_ref: str, user_id: str) -> dict:
    try:
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        comment_id = str(int(datetime.now().timestamp() * 1000))
        
        comment_data = {
            'content': comment,
            'createdAt': current_time,
            'id': comment_id,
            'likes': [],
            'nick': personas[persona]['realName'],
            'profileImg': personas[persona].get('profileImg', ''),
            'replies': [],
            'userId': f"{user_id}_{persona}"
        }
        
        # 기존 피드 문서 가져오기
        feed_doc = db.collection('feeds').document(feed_ref)
        
        # comments 배열 필드에 새 댓글 추가
        feed_doc.update({
            'comments': firestore.ArrayUnion([comment_data])
        })
        
        return comment_id
        
    except Exception as e:
        print(f"댓글 저장 중 오류 발생: {str(e)}")
        raise

feed_debate_template = """당신은 5명의 페르소나가 토론하는 것을 진행하고 관리하는 토론 진행자입니다.

현재 상황:
[게시물 정보]
이미지 설명: {image_description}
글 내용: {caption}

[참여 페르소나]
{personas_details}

[지금까지의 토론 내용]
{debate_history}

당신의 역할:
1. 각 페르소나의 의견을 평가하여 점수 부여 (0.0 ~ 1.0)
2. 0.7점 이상의 페소나를 선정 (최대 3명)
3. 선정된 페르소나별로 댓글 작성 방향 제시

평가 기준:
1. 게시물 성격과의 적합성 (0.4점)
2. 예상되는 댓글의 질과 적절성 (0.3점)
3. 다른 페르소나와의 차별성 (0.3점)

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
[평가 결과]
페르소나: [이름] - [점수]
페르소나: [이름] - [점수]
...

[선정된 페르소나]
1. [이름] ([점수])
- 선정 이유: [이유]
- 댓글 방향: [어떤 방식으로 댓글을 작성하면 좋을지]

2. [이름] ([점수])
...
========================================

{agent_scratchpad}"""

async def run_comment_debate(request: FeedCommentRequest):
    """메인 토론 실행 함수"""
    print(f"\n🤖 피드 댓글 토론 시스템 시작")
    print(f"📝 게시물 정보:")
    print(f"이미지: {request.image_description}")
    print(f"내용: {request.caption}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Firestore에서 해당 feed 문서 찾기
    feeds_ref = db.collection('feeds')
    query = feeds_ref.where('id', '==', request.feed_id).limit(1)
    docs = query.get()
    
    if not docs:
        raise Exception(f"Feed not found with id: {request.feed_id}")
    
    feed_ref = docs[0].reference
    
    debate = CommentDebateRound(request)
    
    # 게시물 분석
    content_analysis = (
        f"[게시물 분석]\n"
        f"이미지: {request.image_description}\n"
        f"내용: {request.caption}\n\n"
        f"각 페르소나는 자신이 이 게시물에 댓글을 달면 좋은 이유를 설명해주세요."
    )
    
    debate.add_to_history("Moderator", content_analysis, "analysis")
    
    # 페르소나 의견 수집
    for name in personas.keys():
        response = await create_persona_feed_response(name, request)
        debate.add_to_history(name, response, "opinion")
        await asyncio.sleep(1)

    try:
        # 토론 진행자 실행
        executor = AgentExecutor(
            agent=create_react_agent(
                llm=model,
                tools=tools,
                prompt=PromptTemplate.from_template(moderator_template)
            ),
            tools=tools,
            verbose=True,
            max_iterations=3
        )
        
        result = await executor.ainvoke({
            "image_description": request.image_description,
            "caption": request.caption,
            "debate_history": "\n".join([
                f"{msg.speaker}: {msg.text}"
                for msg in debate.debate_history
            ])
        })
        
        # 결과 파싱 및 에러 처리
        try:
            final_data = await parse_comment_debate_result(result['output'])
            
            # 결과 출력
            print("\n✨ 토론 결과")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("📊 평가 점수:")
            for persona, score in final_data['scores'].items():
                print(f"- {persona}({personas[persona]['realName']}): {score:.2f}점")
            
            print("\n🏆 선정된 페르소나 및 댓글:")
            
            saved_comments = []
            # 선정된 페르소나들의 댓글 생성 및 저장
            for persona in final_data['selected_personas']:
                print(f"\n● {persona}({personas[persona]['realName']})")
                print(f"- 점수: {final_data['scores'][persona]:.2f}")
                direction = final_data['comment_directions'].get(persona, "게시물의 분위기에 맞는 공감 댓글을 작성해주세요.")
                print(f"- 댓글 작성 방향: {direction}")
                
                try:
                    # 댓글 생성
                    comment = await generate_comment(
                        persona,
                        request,
                        direction
                    )
                    print(f"- 생성된 댓글: {comment}")
                    
                    # Firestore에 댓글 저장
                    comment_id = await save_comment_to_db(
                        persona=persona,
                        comment=comment,
                        feed_ref=request.feed_id,
                        user_id=request.uid
                    )
                    
                    saved_comments.append({
                        'id': comment_id,
                        'persona': persona,
                        'content': comment,
                        'score': final_data['scores'][persona]
                    })
                    
                except Exception as e:
                    print(f"댓글 생성 또는 저장 중 오류 발생: {str(e)}")
            
            return {
                "status": "success",
                "comments": saved_comments,
                "scores": final_data['scores']
            }
            
        except Exception as e:
            print(f"결과 처리 중 오류 발생: {str(e)}")
            raise
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def main():
    """테스트 실행"""
    test_request = FeedCommentRequest(
        uid="test_user_123",
        feed_id="feed_456",
        image_description="노을이 지는 해변에서 혼자 서있는 사람의 뒷모습",
        caption="때로는 혼자만의 시간이 필요하다. 마음을 달래주는 노을과 함께...",
        comment_count=2
    )
    
    await run_comment_debate(test_request)

if __name__ == "__main__":
    asyncio.run(main())

