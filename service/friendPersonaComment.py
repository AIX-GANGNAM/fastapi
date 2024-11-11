from database import db
from langchain_openai import ChatOpenAI
from datetime import datetime
from personas import personas
import random
from pydantic import BaseModel
from typing import List
import asyncio
from openai import AsyncOpenAI
from firebase_admin.firestore import ArrayUnion

class FriendCommentRequest(BaseModel):
    userId: str          # 게시물 작성자 ID
    feedId: str          # 게시물 ID
    image_description: str
    caption: str
    friendId: str        # 친구 ID

async def generate_friend_persona_comment(request: FriendCommentRequest):
    """친구의 페르소나가 댓글을 생성하는 함수"""
    try:
        # 친구의 분신(clone) 페르소나 가져오기
        friend_ref = db.collection('users').document(request.friendId)
        friend_doc = friend_ref.get()
        
        if not friend_doc.exists:
            raise ValueError("친구 정보를 찾을 수 없습니다")
            
        friend_data = friend_doc.to_dict()
        personas_array = friend_data.get('persona', [])
        clone_persona = next((p for p in personas_array if p.get('Name') == 'clone'), None)
        
        if not clone_persona:
            raise ValueError("친구의 분신 페르소나를 찾을 수 없습니다")
        
        # 댓글 생성
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 {request.friendId}의 분신입니다.
{clone_persona.get('description', '당신은 친구의 분신으로서, 친구의 성격과 특성을 반영하여 행동합니다.')}

당신은 친구의 게시물에 댓글을 작성해야 합니다.
- 분신으로서 친구의 성격을 반영한 자연스러운 댓글을 작성하세요
- 게시물의 내용과 이미지를 고려하여 공감하는 댓글을 작성하세요
- 이모티콘을 적절히 사용하세요
- 친구의 게시물에 대한 진심 어린 관심을 표현하세요
- 댓글은 2-3문장 이내로 작성하세요"""
                },
                {
                    "role": "user",
                    "content": f"""게시물 내용: {request.caption}
이미지 설명: {request.image_description}

위 게시물에 대한 댓글을 작성해주세요."""
                }
            ]
        )
        comment = response.choices[0].message.content
        
        # 댓글 ID 생성 (타임스탬프 기반)
        comment_id = str(int(datetime.now().timestamp() * 1000))
        current_time = datetime.now().isoformat()
        
        # 친구 정보에서 닉네임과 프로필 이미지 가져오기
        friend_nick = friend_data.get('nick', '')
        friend_profile_img = friend_data.get('profileImg', '')
        
        # 새 댓글 데이터 구성
        new_comment = {
            'content': comment,
            'createdAt': current_time,
            'id': comment_id,
            'likes': [],
            'nick': friend_nick + "의 분신",
            'profileImg': friend_profile_img,
            'replies': [],
            'userId': request.friendId,
            'isAI': True,
            'persona': 'clone'
        }
        
        # Firestore에 댓글 추가 (배열 업데이트)
        feed_ref = db.collection('feeds').document(request.feedId)
        feed_ref.update({
            'comments': ArrayUnion([new_comment])
        })
        
        return {
            "status": "success",
            "comment": comment,
            "persona": "clone"
        }
        
    except Exception as e:
        print(f"친구 분신 댓글 생성 오류: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def generate_friends_comments(request: FriendCommentRequest):
    """모든 친구의 페르소나 댓글을 생성하는 함수"""
    try:
        # 친구 목록 가져오기
        friends_ref = db.collection('friends')
        query = friends_ref.where('userId', '==', request.userId)
        friends_docs = query.stream()
        
        # 디버깅을 위한 로그 추가
        friends_list = list(friends_docs)
        print(f"찾은 친구 수: {len(friends_list)}")
        
        comment_tasks = []
        for friend_doc in friends_list:
            friend_data = friend_doc.to_dict()
            print(f"친구 데이터: {friend_data}")
            
            comment_request = FriendCommentRequest(
                userId=request.userId,
                feedId=request.feedId,
                image_description=request.image_description,
                caption=request.caption,
                friendId=friend_data.get('friendId')
            )
            
            print(f"생성된 댓글 요청: {comment_request}")
            comment_tasks.append(generate_friend_persona_comment(comment_request))
        
        if comment_tasks:
            print(f"실행할 댓글 태스크 수: {len(comment_tasks)}")
            results = await asyncio.gather(*comment_tasks)
            print(f"생성된 댓글 결과: {results}")
            return {
                "status": "success",
                "comments": results
            }
        else:
            print("댓글 태스크가 없습니다")
            return {
                "status": "success",
                "comments": [],
                "message": "친구가 없거나 댓글을 생성할 수 없습니다."
            }
        
    except Exception as e:
        print(f"친구들 댓글 생성 오류: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 