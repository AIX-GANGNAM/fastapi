from fastapi import HTTPException
from datetime import datetime
import requests
from database import db
from models import NotificationRequest


async def send_expo_push_notification(notification_request: NotificationRequest):  
    print("service > send_expo_push_notification 호출")
    # 푸시 알림 보내기 필요한 정보 (uid, whoSendMessage(누가 보내는지), message(알림 메시지), pushType(어떤 타입으로 알람을 보내는지, chat, persona, follow, like, 등등))
    targetUid = notification_request.targetUid
    fromUid = notification_request.fromUid
    whoSendMessage = notification_request.whoSendMessage
    message = notification_request.message
    screenType = notification_request.screenType
    URL = notification_request.URL
    
    print("누구에게 보내는지 : ", targetUid)
    print("누가 보내는지 : ", fromUid)
    print("메세지 내용 : ", message)
    print("알람 타입 : ", screenType)
    print("URL : ", URL)

    targetUser_ref = db.collection('users').document(targetUid)
    targetUser_doc = targetUser_ref.get()
    if targetUser_doc.exists:
        targetExpo_token = targetUser_doc.to_dict().get('pushToken') # 푸시 토큰 가져오기
        if targetExpo_token:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
                'Content-Type': 'application/json',
            }

            # 푸시 알림 데이터 (JSON)
            # 알람 미리보기에 표시되는 내용 (알림 종류, 누가, 무엇을, 했어요 ) (pushType, whoSendMessage, message)
            # ex) 좋아요 알림, 최창욱님이 당신의 피드에 좋아요를 눌렀어요.
            # ex) 친구요청 알림, 최창욱님이 당신에게 친구요청을 보냈어요.
            payload = {
                "to": targetExpo_token, # 받는 사람의 푸시 토큰
                "sound": 'default', # 알림 소리
                "title": f"{whoSendMessage}", # 알림 제목
                "body": message, # 알림 메시지          
                "priority": "high",          # 알림 우선순위를 높게 설정
                "channelId": 'default', # 알림 채널 아이디
                "data": {
                    "whoSendMessage":  whoSendMessage, # 알림 보내는 사람의 아이디 or 이메일 or 페르소나 이름
                    "highlightTitle": screenType, # 알람이 보여질 때 표시될 제목
                    "fromUid": fromUid, # 알림 보내는 사람의 아이디
                    "highlightImage": 'https://example.com/default-image.jpg', # 알림 보내는 사람 이미지 // 이건 수정 해야 
                    "screenType": screenType, # 어떤 타입으로 알람을 보내는지 (ex. 페르소나 알림, 채팅 알림, 채팅 메시지 알림)
                    "URL": URL, # 알림 클릭 시 이동할 URL
                    "pushTime": datetime.now().isoformat(), # 푸시 알람 시간
                    "priority": "high", # 알림 우선순위를 높게 설정
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