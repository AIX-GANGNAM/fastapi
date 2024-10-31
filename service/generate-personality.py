from models import GeneratePersonalityRequest

async def generate_personality(request: GeneratePersonalityRequest):
    print(f"\n사용자 정의 페르소나 생성 요청:")
    print(f"UID: {request.uid}")
    print(f"이름: {request.name}")
    print(f"성격: {request.personality}")
    print(f"말투: {request.speechStyle}")
    
    # 일단 요청 받았다는 응답만 반환
    return {
        "message": "페르소나 생성 요청이 접수되었습니다",
        "data": {
            "uid": request.uid,
            "name": request.name,
            "personality": request.personality,
            "speechStyle": request.speechStyle
        }
    }