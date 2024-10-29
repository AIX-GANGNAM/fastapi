from fastapi import HTTPException
from datetime import datetime
import requests
from database import db


def send_expo_push_notification(uid: str, whoSendMessage: str, message: str, type: str):
    # 푸시 알림 보내기 필요한 정보 (uid, whoSendMessage(누가 보내는지), message(알림 메시지), type(어떤 타입으로 알람을 보내는지, chat, persona, follow, like, 등등))
    print("service > send_expo_push_notification 호출")
    print("uid : ", uid)
    print("누가 보내는지 : ", whoSendMessage)
    print("메세지 내용 : ", message)
    print("알람 타입 : ", type)

    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()
    if user_doc.exists:
        expo_token = user_doc.to_dict().get('pushToken') # 푸시 토큰 가져오기
        if expo_token:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
                'Content-Type': 'application/json',
            }

            # 푸시 알림 데이터 (JSON)
            payload = {
                "to": expo_token, # 푸시 토큰
                "sound": 'default', # 알림 소리
                "title": f"{whoSendMessage}", # 알림 제목
                "body": message, # 알림 메시지          
                "priority": "high",          # 알림 우선순위를 높게 설정
                "data": {
                    "whoSendMessage": whoSendMessage, # 알림 보내는 사람의 아이디 or 이메일 or 페르소나 이름
                    "highlightTitle": whoSendMessage, # 알림 보내는 사람의 대표 이미지
                    "highlightImage": 'https://example.com/default-image.jpg', # 알림 보내는 사람 이미지
                    "pushType": type, # 어떤 타입으로 알람을 보내는지 (ex. 페르소나 알림, 채팅 알림, 채팅 메시지 알림)
                    "pushTime": datetime.now().isoformat(), # 푸시 알 시간
                },
                
            }
            print("services > send_expo_push_notification > payload : ", payload)

            # Expo 서버로 푸시 알림 요청 전송
            response = requests.post("https://exp.host/--/api/v2/push/send", json=payload, headers=headers)
            print("services > send_expo_push_notification > 전송결과", response)
            # 요청 결과 처리
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        else:
            raise HTTPException(status_code=404, detail="푸시 토큰을 찾을 수 없습니다.")
    else:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    return response.json()