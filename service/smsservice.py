from sdk.api.message import Message
from sdk.exceptions import CoolsmsException
import os
from dotenv import load_dotenv
import json  # json 모듈 추가
import re
import unicodedata

# 환경 변수 로드
load_dotenv()

SMS_API_KEY = os.getenv("SMS_API_KEY")
SMS_API_SECRET = os.getenv("SMS_API_SECRET")
SENDER_NUMBER = os.getenv("SENDER_NUMBER")

# 동기 SMS 전송 로직을 처리하는 서비스 함수
def send_sms_service(request):
    if isinstance(request, str):
        try:
            request = request.replace("\\", "").replace("\n", "").replace("\r", "").strip()
            request = json.loads(request)
        except json.JSONDecodeError as jde:
            print(f"JSON 파싱 오류: {str(jde)}")
            return {"status": "fail", "message": "잘못된 JSON 형식", "status_code": 400}

    # 이모지 및 특수문자 처리
    message = request.get("message", "")
    
    # 1. 이모지 제거
    message = re.sub(r'[^\uAC00-\uD7A3\u0000-\u007F]', '', message)
    
    # 2. 문자열 정규화
    message = unicodedata.normalize('NFC', message)
    
    # 3. SMS 파라미터 설정
    params = {
        "type": "SMS",
        "to": request.get("phone_number"),
        "from": SENDER_NUMBER,
        "text": message,
        "charset": "utf8"
    }

    cool = Message(SMS_API_KEY, SMS_API_SECRET)

    try:
        response = cool.send(params)
        if response["success_count"] > 0:
            print(f"SMS 전송 성공: {request.get('phone_number')}")
            return {"status": "success", "message": message}
        else:
            print(f"SMS 전송 실패: {request.get('phone_number')}")
            return {"status": "fail", "message": "SMS 전송 실패", "status_code": 400}

    except CoolsmsException as e:
        print(f"서버 오류: {e.msg}")
        return {"status": "fail", "message": f"서버 오류: {e.msg}", "status_code": 500}
