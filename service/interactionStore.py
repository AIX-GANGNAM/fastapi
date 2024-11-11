from datetime import datetime
from database import db, redis_client
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
import json

# GPT-4 모델 초기화
gpt4_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 상호작용 분석 프롬프트
analysis_template = """당신은 사용자의 분신 페르소나를 발전시키는 전문가입니다. 
사용자의 최근 대화와 행동 패턴을 분석하여 사용자를 더욱 닮은 분신으로 발전시켜주세요.

현재 분신 페르소나:
tone: {current_tone}
example: {current_example}
description: {current_description}

최근 상호작용 기록:
{interactions}

위 정보를 바탕으로 사용자의 분신을 더욱 발전시켜주세요.
이전 페르소나의 특성을 유지하면서, 새롭게 발견된 사용자의 특성을 자연스럽게 반영해주세요.

다음 형식으로 응답해주세요:

tone: 사용자의 실제 말투와 의사소통 방식을 더욱 자연스럽게 반영. 
- 주로 사용하는 어투
- 감정 표현 방식
- 이모티콘 사용 패턴
- 문장 종결 방식
등을 구체적으로 설명해주세요.

example: 실제 사용자의 대화에서 발견된 특징적인 표현들을 반영한 자연스러운 대화 예시.
반드시 사용자의 실제 대화 스타일을 반영해야 합니다.

personality_traits: 이전 성격 특성에 새롭게 발견된 특성을 자연스럽게 통합하여 설명.
- 기존 성격의 발전된 모습
- 새롭게 발견된 성향
- 상황별 반응 패턴
등을 포함해주세요.

모든 응답은 반드시 한글로 작성하고, 사용자를 그대로 반영하는 분신이 되도록 해주세요."""

# 프롬프트 템플릿 생성
analysis_prompt = PromptTemplate(
    input_variables=["current_tone", "current_example", "current_description", "interactions"],
    template=analysis_template
)

# RunnableSequence 생성
analysis_chain = analysis_prompt | gpt4_model

async def store_user_interaction(uid: str, interaction_type: str, content: str, importance: int = 5):
    """사용자의 상호작용을 Redis에 저장하고, 필요시 분석을 수행합니다."""
    try:
        redis_key = f"user:{uid}:interactions"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 상호작용 데이터 저장
        interaction_data = {
            "timestamp": current_time,
            "message": content,
            "type": interaction_type,  # 'chat' or 'comment'
            "importance": importance
        }
        
        print(f"interaction_data: {interaction_data}")
        # Redis에 저장
        redis_client.lpush(redis_key, json.dumps(interaction_data))
        
        # 저장된 상호작용 수 확인
        interactions_count = redis_client.llen(redis_key)

        print(f"저장된 상호작용 수: {interactions_count}")
        
        # 10개가 쌓이면 분석 및 업데이트 실행
        if interactions_count >= 10:
            await analyze_and_update_persona(uid)
            # 분석 후 Redis 데이터 초기화
            redis_client.delete(redis_key)
            
    except Exception as e:
        print(f"상호작용 저장 중 오류 발생: {str(e)}")

async def analyze_interactions_with_llm(current_clone: dict, interactions: list):
    """LLM을 사용하여 상호작용을 분석합니다."""
    try:
        # 상호작용 데이터 포맷팅
        interaction_text = "\n".join([
            f"[{item['timestamp']}] ({item['type']}) {item['message']}"
            for item in interactions
        ])
        
        # LLM 분석 실행
        result = analysis_chain.invoke({
            "current_tone": current_clone.get('tone', ''),
            "current_example": current_clone.get('example', ''),
            "current_description": current_clone.get('description', ''),
            "interactions": interaction_text
        })
        
        # 결과 파싱 (ChatOpenAI의 출력 형식에 맞게 수정)
        response_text = result.content if hasattr(result, 'content') else str(result)
        sections = {}
        current_section = None
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line:
                if line.startswith('tone:'):
                    current_section = 'tone'
                    sections[current_section] = line.replace('tone:', '').strip()
                elif line.startswith('example:'):
                    current_section = 'example'
                    sections[current_section] = line.replace('example:', '').strip()
                elif line.startswith('personality_traits:'):
                    current_section = 'personality_traits'
                    sections[current_section] = line.replace('personality_traits:', '').strip()
                elif current_section:
                    sections[current_section] += ' ' + line
        
        return sections
        
    except Exception as e:
        print(f"상호작용 분석 중 오류 발생: {str(e)}")
        return {}

async def analyze_and_update_persona(uid: str):
    """저장된 상호작용을 분석하고 페르소나를 업데이트합니다."""
    try:
        redis_key = f"user:{uid}:interactions"
        interactions = []
        
        # 모든 상호작용 데이터 가져오기
        for i in range(redis_client.llen(redis_key)):
            interaction = redis_client.lindex(redis_key, i)
            if interaction:
                interactions.append(json.loads(interaction))
        
        # 현재 페르소나 정보 가져오기
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            personas = user_doc.to_dict().get('persona', [])
            current_clone = next((p for p in personas if p.get('Name') == 'clone'), None)
            
            if not current_clone:
                return
            
            # LLM 분석 실행
            analysis_result = await analyze_interactions_with_llm(current_clone, interactions)
            
            if not analysis_result:
                return
            
            # clone 페르소나 업데이트
            updated_personas = []
            for persona in personas:
                if persona.get('Name') == 'clone':
                    # 기존 데이터 유지하면서 분석 결과 반영
                    updated_persona = persona.copy()
                    updated_persona.update({
                        "tone": analysis_result.get('tone', persona.get('tone')),
                        "example": analysis_result.get('example', persona.get('example')),
                        "description": analysis_result.get('personality_traits', persona.get('description'))
                    })
                    updated_personas.append(updated_persona)
                else:
                    updated_personas.append(persona)
            
            # Firestore 업데이트
            user_ref.update({
                'persona': updated_personas
            })
            
            print(f"사용자 {uid}의 페르소나가 성공적으로 업데이트되었습니다.")
            
    except Exception as e:
        print(f"페르소나 분석 및 업데이트 중 오류 발생: {str(e)}") 