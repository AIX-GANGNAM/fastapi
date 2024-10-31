from models import UserProfile
from database import db
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# GPT-4 모델 초기화
gpt4_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 프로필 기반 페르소나 생성 프롬프트
profile_persona_template = """당신은 사용자의 프로필 정보를 바탕으로 자연스러운 페르소나를 생성하는 전문가입니다.

사용자 프로필:
MBTI: {mbti}
성격: {personality}
관심사: {interests}
대화 스타일: {communication_style}
말투: {speaking_style}
이모티콘 사용: {emoji_style}
가치관: {values}
의사결정 스타일: {decision_style}

이 정보를 바탕으로 자연스러운 페르소나를 생성해주세요.

다음 형식에 맞춰 응답해주세요:

description: 페르소나의 성격과 특성을 2-3문장으로 자세히 설명. MBTI 특성과 가치관, 의사결정 방식을 포함해서 설명.

tone: 페르소나의 말투와 의사소통 스타일을 구체적으로 설명. 대화 스타일과 이모티콘 사용 습관을 반영하여 설명.

example: 페르소나의 관심사와 성격이 드러나는 대화 예시를 작성. 지정된 말투와 이모티콘 스타일을 반영.

응답은 반드시 위 형식을 지켜주시고, 각 항목은 한글로 작성해주세요."""

# 프롬프트 템플릿 생성
profile_prompt = PromptTemplate(
    input_variables=["mbti", "personality", "interests", "communication_style", 
                    "speaking_style", "emoji_style", "values", "decision_style"],
    template=profile_persona_template
)

# 새로운 방식으로 체인 생성
profile_chain = profile_prompt | gpt4_model

async def update_clone_personality(profile: UserProfile):
    try:
        user_ref = db.collection('users').document(profile.uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise Exception("사용자를 찾을 수 없습니다.")
        
        user_data = user_doc.to_dict()
        personas = user_data.get('persona', [])
        user_name = user_data.get('userName', '사용자')
        
        # LLM을 통한 페르소나 생성
        result = profile_chain.invoke({
            "mbti": profile.mbti,
            "personality": ", ".join(profile.personality),
            "interests": ", ".join(profile.interests),
            "communication_style": profile.communication_style,
            "speaking_style": profile.speaking_style,
            "emoji_style": profile.emoji_style,
            "values": ", ".join(profile.values),
            "decision_style": profile.decision_style
        })
        
        # 결과 파싱
        response_text = result['text'].strip()
        sections = {}
        current_section = None
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line:
                if line.startswith('description:'):
                    current_section = 'description'
                    sections[current_section] = line.replace('description:', '').strip()
                elif line.startswith('tone:'):
                    current_section = 'tone'
                    sections[current_section] = line.replace('tone:', '').strip()
                elif line.startswith('example:'):
                    current_section = 'example'
                    sections[current_section] = line.replace('example:', '').strip()
                elif current_section:
                    sections[current_section] += ' ' + line
        
        # 기존 페르소나 업데이트
        updated_personas = []
        clone_updated = False
        
        for persona in personas:
            if persona.get('Name') == 'clone':
                # 기존 persona 객체를 유지하면서 필요한 필드만 업데이트
                updated_persona = persona.copy()  # 기존 데이터 복사
                updated_persona.update({          # 필요한 필드만 업데이트
                    "DPNAME": f"{user_name}의 분신",
                    "description": sections.get('description', ''),
                    "tone": sections.get('tone', ''),
                    "example": sections.get('example', '')
                })
                updated_personas.append(updated_persona)
                clone_updated = True
            else:
                updated_personas.append(persona)
        
        if not clone_updated:
            # 새로운 clone 생성 시에만 전체 필드 설정
            new_clone = {
                "Name": "clone",
                "DPNAME": f"{user_name}의 분신",
                "description": sections.get('description', ''),
                "tone": sections.get('tone', ''),
                "example": sections.get('example', '')
            }
            updated_personas.append(new_clone)
        
        # Firestore 업데이트
        user_ref.update({
            'persona': updated_personas
        })
        
        return {
            "message": "분신 페르소나가 성공적으로 업데이트되었습니다",
            "data": updated_personas[-1]  # 업데이트된 clone 반환
        }
        
    except Exception as e:
        print(f"프로필 업데이트 중 오류 발생: {str(e)}")
        return {
            "message": "프로필 업데이트 중 오류가 발생했습니다",
            "error": str(e)
        }