import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import chromadb
import redis
from datetime import datetime

load_dotenv()

# Firebase 초기화
cred = credentials.Certificate("mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'mirrorgram-20713.appspot.com'
})
db = firestore.client()

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path="./chroma_db")

# OpenAI API 키 설정
from openai import OpenAI
aiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_persona_collection(uid, persona_name):
    return client.get_or_create_collection(f"{uid}_inside_out_persona_{persona_name}")

def get_user_collection(uid):
    """사용자별 단일 컬렉션 생성 또는 가져오기"""
    return client.get_or_create_collection(f"user_{uid}_memories")

def store_long_term_memory(uid: str, persona_name: str, memory: str, memory_type: str):
    """벡터 DB에 통합 메모리 저장"""
    collection = get_user_collection(uid)

    # 임베딩 생성
    embedding = aiclient.embeddings.create(
        input=memory,
        model="text-embedding-ada-002"
    ).data[0].embedding

    # 메타데이터 구성
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "type": memory_type,  # 파라미터로 받은 타입 사용
        "persona_name": persona_name,
    }

    # 고유 ID 생성
    unique_id = f"{uid}_{metadata['type']}_{metadata['persona_name']}_{metadata['timestamp']}"

    # 컬렉션에 저장
    collection.add(
        documents=[memory],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]
    )

def query_memories(
    uid: str, 
    query: str, 
    memory_type: str = None,
    persona_name: str = None,
    limit: int = 5
):
    """메모리 검색 함수"""
    collection = get_user_collection(uid)
    
    # 쿼리 임베딩 생성
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    
    # 필터 조건 구성
    where_conditions = {}
    if memory_type:
        where_conditions["type"] = memory_type
    if persona_name:
        where_conditions["persona_name"] = persona_name
    
    try:
        # 검색 실행
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_conditions if where_conditions else None
        )
        return results
    except Exception as e:
        print(f"메모리 검색 오류: {str(e)}")
        # 검색 실패 시 빈 결과 반환
        return {
            "documents": [[]],
            "distances": [],
            "metadatas": []
        }

redis_client = redis.Redis(host='localhost', port=6379, db=0)