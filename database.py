import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import chromadb
import redis

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

redis_client = redis.Redis(host='redis', port=6379, db=0)