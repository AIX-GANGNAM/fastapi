from fastapi import HTTPException
from datetime import datetime
import requests
from database import db
from models import NotificationRequest
import asyncio

async def send_expo_push_notification(notification_request: NotificationRequest):  
    print("service > send_expo_push_notification 호출")
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
    
    if not targetUser_doc.exists:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        
    user_data = targetUser_doc.to_dict()
    push_tokens = user_data.get('pushTokens', [])
    
    # 이전 버전 호환성을 위한 처리
    if old_token := user_data.get('pushToken'):
        if old_token not in push_tokens:
            push_tokens.append(old_token)
    
    if not push_tokens:
        raise HTTPException(status_code=404, detail="푸시 토큰을 찾을 수 없습니다.")

    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Content-Type': 'application/json',
    }

    base_payload = {
        "sound": 'default',
        "title": f"{whoSendMessage}",
        "body": message,
        "priority": "high",        
        "channelId": 'channel_high',
        "data": {
            "whoSendMessage": whoSendMessage,
            "highlightTitle": screenType,
            "fromUid": fromUid,
            "highlightImage": 'https://example.com/default-image.jpg',
            "screenType": screenType,
            "URL": URL,
            "pushTime": datetime.now().isoformat(),
        }
    }

    async def send_single_notification(token):
        try:
            payload = {**base_payload, "to": token}
            print(f"토큰으로 알림 전송 시도: {token}")
            response = requests.post(
                "https://exp.host/--/api/v2/push/send", 
                json=payload, 
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"알림 전송 성공: {token} + {datetime.now()}")
                return {"token": token, "success": True, "response": response.json()}
            else:
                print(f"알림 전송 실패: {token}, 상태 코드: {response.status_code}")
                return {"token": token, "success": False, "error": response.text}
                
        except Exception as e:
            print(f"알림 전송 중 오류 발생: {token}, 오류: {str(e)}")
            return {"token": token, "success": False, "error": str(e)}

    # 모든 토큰에 대해 비동기로 알림 전송
    notification_tasks = [send_single_notification(token) for token in push_tokens]
    results = await asyncio.gather(*notification_tasks)
    
    # 실패한 토큰 처리 (선택적)
    failed_tokens = [result["token"] for result in results if not result["success"]]
    if failed_tokens:
        print(f"실패한 토큰들: {failed_tokens}")
        # 여기서 필요하다면 실패한 토큰을 데이터베이스에서 제거하는 로직을 추가할 수 있습니다
        
    # 하나라도 성공했다면 성공으로 간주
    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        raise HTTPException(status_code=500, detail="모든 알림 전송이 실패했습니다.")
    
    return {
        "success": True,
        "total_tokens": len(push_tokens),
        "successful_deliveries": len(successful_results),
        "failed_deliveries": len(failed_tokens),
        "results": results
    }