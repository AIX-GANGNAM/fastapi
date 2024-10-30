from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from models import GeneratePersonalityRequest
from database import db  # database.py에서 Firestore 클라이언트 import

# GPT-4 모델 초기화
gpt4_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 페르소나 생성 프롬프트 템플릿
persona_template = """당신은 페르소나의 특성을 상세하게 설명하는 전문가입니다.
주어진 이름, 성격, 말투를 바탕으로 페르소나의 상세 프로필을 생성해주세요.

입력 정보:
이름: {name}
성격: {personality}
말투: {speechStyle}

다음 형식에 맞춰 응답해주세요:

description: 페르소나의 성격과 특성을 2-3문장으로 자세히 설명. 장단점을 포함하고 다른 사람들과의 관계나 영향도 언급.

tone: 페르소나의 말투와 의사소통 스타일을 구체적으로 설명. 반드시 반말을 사용하며, 자주 사용하는 어투나 특징적인 표현 방식을 포함.

example: 반드시 반말로 된 대화 예시를 작성. 예: "야 너도 이거 해볼래? 진짜 재밌어!😊" 처럼 이모티콘과 함께 친근한 반말을 사용.

응답은 반드시 위 형식을 지켜주시고, 각 항목은 한글로 작성해주세요. 특히 tone과 example은 꼭 반말로 작성해주세요."""

# 프롬프트 템플릿 생성
persona_prompt = PromptTemplate(
    input_variables=["name", "personality", "speechStyle"],
    template=persona_template
)

# LLM 체인 생성
persona_chain = LLMChain(llm=gpt4_model, prompt=persona_prompt)

async def generate_personality(request: GeneratePersonalityRequest):
    print(f"\n사용자 정의 페르소나 생성 요청:")
    print(f"UID: {request.uid}")
    print(f"이름: {request.name}")
    print(f"성격: {request.personality}")
    print(f"말투: {request.speechStyle}")
    
    try:
        # LLM 체인 실행
        result = persona_chain.invoke({
            "name": request.name,
            "personality": request.personality,
            "speechStyle": request.speechStyle
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
        
        # Firestore에서 현재 사용자의 persona 배열 가져오기
        user_ref = db.collection('users').document(request.uid)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            current_personas = user_data.get('persona', [])
            
            # 사용자 정의 페르소나 생성
            new_persona = {
                "Name": "custom",
                "DPNAME": request.name,
                "description": sections.get('description', ''),
                "tone": sections.get('tone', ''),
                "example": sections.get('example', '')
            }
            
            # 분신 페르소나 생성
            user_name = user_data.get('profile', {}).get('userName', '사용자')
            clone_result = await generate_clone_data(user_name)
            clone_persona = {
                "Name": "clone",
                "DPNAME": f"{user_name}의 분신",
                "description": clone_result.get('description', ''),
                "tone": clone_result.get('tone', ''),
                "example": clone_result.get('example', '')
            }
            
            # 기존 페르소나 업데이트 또는 추가
            updated_personas = []
            custom_exists = False
            clone_exists = False
            
            for persona in current_personas:
                if persona.get('Name') == 'custom':
                    updated_personas.append(new_persona)
                    custom_exists = True
                elif persona.get('Name') == 'clone':
                    updated_personas.append(clone_persona)
                    clone_exists = True
                else:
                    updated_personas.append(persona)
            
            # 없는 경우 새로 추가
            if not custom_exists:
                updated_personas.append(new_persona)
            if not clone_exists:
                updated_personas.append(clone_persona)
            
            # Firestore 업데이트
            user_ref.update({
                'persona': updated_personas
            })
            
            print(f"페르소나와 분신이 성공적으로 저장되었습니다")
        else:
            # 사용자 문서가 없는 경우 새로 생성
            user_ref.set({
                'persona': [new_persona, clone_persona]
            })
            print(f"새로운 사용자 문서가 생성되었습니다: {request.uid}")
        
        return {
            "message": "페르소나가 성공적으로 생성되었습니다",
            "data": {
                "uid": request.uid,
                "name": request.name,
                "description": sections.get('description', ''),
                "tone": sections.get('tone', ''),
                "example": sections.get('example', '')
            }
        }
        
    except Exception as e:
        print(f"페르소나 생성 중 오류 발생: {str(e)}")
        return {
            "message": "페르소나 생성 중 오류가 발생했습니다",
            "error": str(e)
        }
    
# 분신 데이터 생성을 위한 헬퍼 함수
async def generate_clone_data(user_name: str):
    clone_personality = f"{user_name}의 특성을 반영한 분신으로, 사용자와 비슷한 성향을 가지고 있지만 독립적인 개성도 지님"
    clone_speech = "사용자와 비슷하면서도 약간의 개성이 있는 말투"
    
    result = persona_chain.invoke({
        "name": f"{user_name}의 분신",
        "personality": clone_personality,
        "speechStyle": clone_speech
    })
    
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
    
    return sections
    
