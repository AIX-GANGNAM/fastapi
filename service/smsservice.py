from sdk.api.message import Message
from sdk.exceptions import CoolsmsException
import os
from dotenv import load_dotenv
import json  # json 모듈 추가

# 환경 변수 로드
load_dotenv()

SMS_API_KEY = os.getenv("SMS_API_KEY")
SMS_API_SECRET = os.getenv("SMS_API_SECRET")
SENDER_NUMBER = os.getenv("SENDER_NUMBER")

# 동기 SMS 전송 로직을 처리하는 서비스 함수
def send_sms_service(request):
    # 요청이 문자열인 경우 JSON으로 변환 및 이스케이프 문자 제거
    if isinstance(request, str):
        try:
            request = request.replace("\\", "").replace("\n", "").replace("\r", "").strip()  # 이스케이프 문자 제거
            request = json.loads(request)  # JSON 문자열을 딕셔너리로 변환
        except json.JSONDecodeError as jde:
            print(f"JSON 파싱 오류: {str(jde)}")
            return {"status": "fail", "message": "올바른 JSON 형식이 아닙니다.", "status_code": 400}

    # SMS 전송을 위한 파라미터 설정
    params = {
        "type": "sms",
        "to": request.get("phone_number"),  # JSON 객체에서 값 추출
        "from": SENDER_NUMBER,
        "text": request.get("message"),  # 사용자가 입력한 메시지
    }

    cool = Message(SMS_API_KEY, SMS_API_SECRET)

    try:
        # 동기적으로 SMS 전송 시도
        response = cool.send(params)
        if response["success_count"] > 0:
            print(f"SMS 전송 성공: {request.get('phone_number')}")
            return {"status": "success", "message": "SMS 전송 성공"}
        else:
            print(f"SMS 전송 실패: {request.get('phone_number')}")
            return {"status": "fail", "message": "SMS 전송 실패", "status_code": 400}

    except CoolsmsException as e:
        print(f"서버 오류: {e.msg}")
        return {"status": "fail", "message": f"서버 오류: {e.msg}", "status_code": 500}
