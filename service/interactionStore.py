from database import db
from google.cloud import firestore
from datetime import datetime
import json

async def store_user_interaction(uid: str, message: str, interaction_type: str = 'chat'):
    """사용자 상호작용 저장"""
    try:
        # Firestore에 상호작용 저장
        interactions_ref = db.collection('users').document(uid).collection('interactions')
        
        interaction_data = {
            'message': message,
            'type': interaction_type,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # 상호작용 저장
        interactions_ref.add(interaction_data)
        
        # 상호작용 수 업데이트
        user_ref = db.collection('users').document(uid)
        user_ref.update({
            'interactionCount': firestore.Increment(1)
        })
        
        # 저장된 상호작용 수 확인 (디버깅용)
        interactions = interactions_ref.where(
            'date', '==', interaction_data['date']
        ).get()
        print(f"저장된 상호작용 수: {len(list(interactions))}")
        
        return True
        
    except Exception as e:
        print(f"상호작용 저장 오류: {str(e)}")
        # 실패해도 메인 기능은 계속 동작하도록 False 반환
        return False 