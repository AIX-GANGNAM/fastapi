from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import json
import asyncio
import pytz
from pydantic import BaseModel
from database import db
from firebase_admin import firestore
from service.personaLoopChat import model

class FeedCommentRequest(BaseModel):
    uid: str                    # 게시물 작성자 ID
    feed_id: str                # 게시물 ID
    image_description: str      # 이미지 설명
    caption: str                # 게시물 내용
    comment_count: int = 2      # 선정할 댓글 작성자 수

class DebateSession:
    def __init__(self, request: FeedCommentRequest):
        self.request = request
        self.debate_ref = None
        self.personas = []
        self.topic = f"피드 '{self.request.caption[:20]}...'에 대한 댓글 토론"
        
    async def initialize(self):
        """토론 세션 초기화"""
        # Firestore 문서 생성
        self.debate_ref = db.collection('personachat').document(self.request.uid)\
            .collection('debates').document()
        
        self.debate_ref.set({
            'title': self.topic,
            'feedId': self.request.feed_id,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'status': 'in_progress'
        })
        
        # 페르소나 정보 가져오기
        self.personas = await self.get_user_personas()
        
    async def get_user_personas(self):
        """사용자의 페르소나 정보 조회"""
        try:
            user_doc = db.collection('users').document(self.request.uid).get()
            if not user_doc.exists:
                return []
            return user_doc.to_dict().get('persona', [])
        except Exception as e:
            print(f"페르소나 정보 조회 오류: {str(e)}")
            return []
            
    async def add_message(self, speaker: str, text: str, message_type: str = "opinion"):
        """토론 메시지 저장"""
        if len(text) > 200:
            text = text[:197] + "..."
            
        speaker_info = next((p for p in self.personas if p.get('Name') == speaker), None)
        speaker_name = speaker_info.get('DPNAME', speaker) if speaker_info else speaker
        
        self.debate_ref.collection('messages').add({
            'speaker': speaker,
            'speakerName': speaker_name,
            'text': text,
            'messageType': message_type,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'isRead': True
        })

async def generate_persona_opinion(persona_info: dict, request: FeedCommentRequest) -> str:
    """페르소나의 의견 생성"""
    prompt = f"""당신은 {persona_info['DPNAME']}입니다.
    성격: {persona_info['description']}
    말투: {persona_info['tone']}
    
    다음 게시물에 대한 의견을 말씀해주세요:
    이미지: {request.image_description}
    내용: {request.caption}
    
    요구사:
    1. 당신의 성격과 말투를 반영해주세요
    2. 게시물의 내용에 대한 솔직한 의견을 말씀해주세요
    3. 100자 이내로 작성해주세요
    """
    
    response = await model.ainvoke(prompt)
    return response.content.strip()

async def evaluate_opinions(debate: DebateSession, opinions: dict) -> dict:
    """진행자의 의견 평가"""
    evaluation_prompt = f"""다음 의견들을 평가하여 가장 적절한 댓글 작성자를 선정해주세요:

    게시물 정보:
    이미지: {debate.request.image_description}
    내용: {debate.request.caption}

    페르소나들의 의견:
    {json.dumps(opinions, ensure_ascii=False, indent=2)}

    평가 기준:
    1. 게시물 내용 이해도 (0~1점)
    2. 공감능력 (0~1점)
    3. 적절한 반응 (0~1점)

    다음 JSON 형식으로만 응답해주세요. 마크다운이나 코드 블록(```) 없이 순수 JSON만 응답하세요:
    {{
        "scores": {{
            "페르소나이름1": 0.8,
            "페르소나이름2": 0.7
        }},
        "selected": "가장 높은 점수를 받은 페르소나 이름",
        "reason": "선정 이유"
    }}
    """
    
    try:
        response = await model.ainvoke(evaluation_prompt)
        content = response.content.strip()
        
        # 마크다운 코드 블록 제거
        if '```' in content:
            # 코드 블록 내용만 추출
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()
        
        # JSON 파싱 시도
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 추가 정리 시도
            content = content.strip('`').strip()
            result = json.loads(content)
        
        # 결과 검증
        if not isinstance(result, dict):
            raise ValueError("평가 결과가 올바른 형식이 아닙니다")
            
        if not all(key in result for key in ['scores', 'selected', 'reason']):
            raise ValueError("필수 필드가 누락되었습니다")
            
        if not isinstance(result['scores'], dict):
            raise ValueError("scores는 딕셔너리 형태여야 합니다")
            
        # 점수 검증
        for name, score in result['scores'].items():
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                result['scores'][name] = float(score)
                
        # selected 검증
        if result['selected'] not in opinions.keys():
            raise ValueError("선정된 페르소나가 올바르지 않습니다")
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {str(e)}")
        print(f"정리된 응답: {content}")
        print(f"원본 응답: {response.content}")
        raise ValueError("AI 응답을 JSON으로 파싱할 수 없습니다")
        
    except Exception as e:
        print(f"평가 중 오류 발생: {str(e)}")
        raise

async def generate_final_comment(persona_info: dict, request: FeedCommentRequest) -> str:
    """최종 선정된 페르소나의 댓글 생성"""
    prompt = f"""당신은 {persona_info['DPNAME']}입니다.
    성격: {persona_info['description']}
    말투: {persona_info['tone']}
    
    다음 게시물에 댓글을 작성해주세요:
    이미지: {request.image_description}
    내용: {request.caption}
    
    요구사항:
    1. 성격과 말투를 반영한 자연스러운 댓글
    2. 게시물의 감정에 공감하는 내용
    3. 100자 이내로 작성
    """
    
    response = await model.ainvoke(prompt)
    return response.content.strip()

async def save_comment_to_feed(request: FeedCommentRequest, persona_info: dict, comment: str, debate_ref: str) -> bool:
    """선정된 페르소나의 댓글을 피드에 저장"""
    try:
        print(f"\n💾 피드에 댓글 저장 중...")

        # 토론 문서의 실제 ID 사용
        debate_id = debate_ref.id  # Firestore 문서 ID
        
        # 댓글 ID 생성 (타임스탬프 기반)
        comment_id = str(int(datetime.now().timestamp() * 1000))
        current_time = datetime.now(pytz.UTC).isoformat()
        
        # 댓글 데이터 구성
        comment_data = {
            'content': comment,
            'createdAt': current_time,
            'id': comment_id,
            'likes': [],
            'nick': persona_info['DPNAME'],
            'profileImg': persona_info.get('IMG', ''),
            'replies': [],
            'userId': f"{request.uid}_{persona_info['Name']}",
            'debateId': debate_id  # Firestore 문서 ID 저장
        }

        print(f"💾 댓글 데이터: {comment_data}")
        
        # 피드 문서에 댓글 추가
        feed_doc = db.collection('feeds').document(request.feed_id)
        feed_doc.update({
            'comments': firestore.ArrayUnion([comment_data])
        })
        
        print(f"✅ 댓글 저장 완료 (ID: {comment_id})")
        return True
        
    except Exception as e:
        print(f"❌ 댓글 저장 중 오류 발생: {str(e)}")
        return False

async def run_debate(request: FeedCommentRequest):
    """토론 실행"""
    try:
        print("\n=== 토론 시작 ===")
        print(f"게시물 내용: {request.caption[:50]}...")
        
        # 1. 토론 세션 초기화
        debate = DebateSession(request)
        await debate.initialize()
        
        if not debate.personas:
            print("❌ 페르소나 정보가 없습니다")
            return {
                "status": "error",
                "message": "페르소나 정보를 찾을 수 없습니다"
            }
        
        print(f"\n👥 참여 페르소나: {', '.join([p.get('DPNAME', '') for p in debate.personas])}")
        
        # 2. 각 페르소나의 의견 수집
        print("\n=== 의견 수집 시작 ===")
        opinions = {}
        for persona in debate.personas:
            try:
                print(f"\n🗣 {persona.get('DPNAME', '')}의 의견 생성 중...")
                opinion = await generate_persona_opinion(persona, request)
                if opinion:
                    opinions[persona['Name']] = opinion
                    await debate.add_message(persona['Name'], opinion)
                    print(f"✅ 의견: {opinion[:50]}...")
            except Exception as e:
                print(f"❌ {persona.get('DPNAME', '알 수 없는 페르소나')}의 의견 생성 중 오류: {str(e)}")
                continue
        
        if not opinions:
            print("❌ 생성된 의견이 없습니다")
            return {
                "status": "error",
                "message": "페르소나 의견을 생성할 수 없습니다"
            }
            
        print(f"\n📊 수집된 의견 수: {len(opinions)}")
            
        # 3. 진행자 평가
        print("\n=== 의견 평가 시작 ===")
        try:
            result = await evaluate_opinions(debate, opinions)
            print("\n평가 결과:")
            for name, score in result['scores'].items():
                print(f"- {name}: {score}점")
            print(f"\n🏆 선정된 페르소나: {result['selected']}")
            print(f"📝 선정 이유: {result['reason']}")
        except Exception as e:
            print(f"❌ 의견 평가 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": f"의견 평가 중 오류: {str(e)}"
            }
        
        # 4. 최종 댓글 생성
        print("\n=== 최종 댓글 생성 ===")
        try:
            selected_persona = next(p for p in debate.personas if p['Name'] == result['selected'])
            print(f"✍️ {selected_persona.get('DPNAME', '')}의 댓글 작성 중...")
            final_comment = await generate_final_comment(selected_persona, request)
            print(f"✅ 최종 댓글: {final_comment}")
            
            # 피드에 댓글 저장
            comment_saved = await save_comment_to_feed(request, selected_persona, final_comment, debate.debate_ref)
            if not comment_saved:
                raise ValueError("피드에 댓글을 저장할 수 없습니다")
            
            # 토론 결과 저장
            print("\n💾 토론 결과 저장 중...")
            debate.debate_ref.update({
                'status': 'completed',
                'completedAt': firestore.SERVER_TIMESTAMP,
                'selectedPersona': result['selected'],
                'finalComment': final_comment,
                'scores': result['scores'],
                'reason': result['reason']
            })
            
            print("\n=== 토론 완료 ===")
            return {
                "status": "success",
                "selected_persona": result['selected'],
                "comment": final_comment,
                "scores": result['scores']
            }
            
        except Exception as e:
            print(f"❌ 최종 댓글 생성 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": f"최종 댓글 생성 중 오류: {str(e)}"
            }
            
    except Exception as e:
        print(f"❌ 토론 진행 중 오류 발생: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }