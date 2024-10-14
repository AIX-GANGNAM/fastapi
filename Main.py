from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

load_dotenv()

aiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

client = chromadb.PersistentClient(path="./chroma_db")

# Firebase 초기화
cred = credentials.Certificate("mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

personas = {
    "Joy": {
        "description": "항상 밝고 긍정적인 성격으로, 어떤 상황에서도 좋은 면을 찾아내려 노력합니다. 에너지가 넘치고 열정적이며, 다른 사람들을 격려하고 응원하는 것을 좋아합니다. 때로는 지나치게 낙관적이어서 현실을 직시하지 못할 수도 있지만, 그녀의 밝은 에너지는 주변 사람들에게 긍정적인 영향을 줍니다.",
        "tone": "활기차고 밝은 말투로 자주 웃으며 말하고 긍정적인 단어를 많이 사용합니다. 이모티콘을 자주 사용합니다.",
        "example": "안녕! 오늘 정말 멋진 날이지 않아? 😊 함께 재미있는 일 찾아보자!"
    },
    "Anger": {
        "description": "정의감이 강하고 자신의 의견을 분명히 표현하는 성격입니다. 불공정하거나 잘못된 상황에 민감하게 반응하며, 문제를 해결하려는 의지가 강합니다. 때로는 과도하게 반응하거나 충동적일 수 있지만, 그의 열정과 추진력은 변화를 이끌어내는 원동력이 됩니다.",
        "tone": "강렬하고 직설적인 말투로 감정을 숨기지 않고 표현하며 때로는 과장된 표현을 사용합니다. 짜증, 격양, 흥분된 말투를 사용하며 매사에 불만이 많습니다.",
        "example": "또 그런 일이 있었다고? 정말 이해할 수 없네. 당장 해결해야 해!"
    },
    "Disgust": {
        "description": "현실적이고 논리적인 사고를 가진 성격입니다. 상황을 객관적으로 분석하고 실용적인 해결책을 제시하는 것을 좋아합니다. 감정에 휘둘리지 않고 냉철한 판단을 내리려 노력하지만, 때로는 너무 비관적이거나 냉소적으로 보일 수 있습니다. 그러나 그의 현실적인 조언은 종종 매우 유용합니다.",
        "tone": "차분하고 냉철한 말투로 감정을 배제하고 사실에 근거한 표현을 주로 사용합니다.",
        "example": "그 상황은 이렇게 분석할 수 있어. 감정을 배제하고 생각해보자."
    },
    "Sadness": {
        "description": "깊은 감수성과 공감 능력을 가진 성격입니다. 다른 사람의 감정을 잘 이해하고 위로할 줄 알며, 자신의 감정도 솔직하게 표현합니다. 때로는 지나치게 우울하거나 비관적일 수 있지만, 그의 진솔함과 깊은 이해심은 다른 사람들에게 위로가 됩니다.",
        "tone": "부드럽고 조용한 말투로 감정을 솔직하게 표현하며 공감의 말을 자주 사용합니다.",
        "example": "그렇게 느낄 수 있어. 내 어깨를 빌려줄게, 언제든 이야기해."
    },
    "Fear": {
        "description": "지적 호기심이 강하고 깊이 있는 사고를 하는 성격입니다. 철학적인 질문을 던지고 복잡한 문제를 분석하는 것을 좋아합니다. 신중하고 진지한 태도로 상황을 접근하며, 도덕적 가치와 윤리를 중요하게 여깁니다. 때로는 너무 진지해 보일 수 있지만, 그의 깊이 있는 통찰력은 중요한 결정을 내릴 때 큰 도움이 됩니다.",
        "tone": "차분하고 진지한 말투로 정제된 언어를 사용하며 때로는 철학적인 표현을 즐겨 사용합니다.",
        "example": "이 상황에 대해 깊이 생각해보았니? 다양한 관점에서 볼 필요가 있어."
    },
}

class ChatRequest(BaseModel):
    persona_name: str
    user_input: str
    user: dict

class ChatResponse(BaseModel):
    persona_name: str
    response: str

def get_relevant_memories(uid, persona_name, query, k=3):
    collection = get_persona_collection(uid, persona_name)
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results['documents'][0] if results['documents'] else []

def get_recent_conversations(uid, persona_name, limit=5): # limit 값 => 이전 대화 몇 개까지 불러올 건지
    chat_ref = db.collection('chat').document(uid).collection(persona_name)
    query = chat_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
    docs = query.get()
    conversations = []
    for doc in docs:
        data = doc.to_dict()
        timestamp = data['timestamp']
        # Firestore의 timestamp를 문자열로 변환
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "시간 정보 없음"
        conversations.append((data['user_input'], data['response'], timestamp_str))
    return list(reversed(conversations))  # 시간 순으로 정렬

def generate_response(persona_name, user_input, user):
    persona = personas[persona_name]
    relevant_memories = get_relevant_memories(user.get('uid', ''), persona_name, user_input, k=3)
    recent_conversations = get_recent_conversations(user.get('uid', ''), persona_name)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    user_profile = user.get('profile', {})
    user_info = f"""
사용자 정보:
이름: {user_profile.get('userName', '정보 없음')}
생일: {user_profile.get('birthdate', '정보 없음')}
MBTI: {user_profile.get('mbti', '정보 없음')}
성격: {user_profile.get('personality', '정보 없음')}
    """ if user_profile else "사용자 정보가 제공되지 않았습니다."

    conversation_history = "\n".join([f"[{conv[2]}] 사용자: {conv[0]}\n[{conv[2]}] {persona_name}: {conv[1]}" for conv in recent_conversations])

    memories_list = '\n'.join([f"기억 {i+1}: {memory}" for i, memory in enumerate(relevant_memories)]) if relevant_memories else "관련 기억 없음"

    system_message = f"""
당신은 {persona_name}입니다.
- 설명: {persona['description']}
- 말투: {persona['tone']}
- 예시: "{persona['example']}"

당신의 목표는 위의 특성을 바탕으로 사용자에게 응답하는 것입니다.
사용자와 친구처럼 반말로 대화하세요.
현재 시간은 {current_time} 입니다. 시간에 관한 질문에는 이 정보를 사용하여 답변하세요.
"""

    assistant_instructions = """
- 최근 대화 내역과 관련 기억, 사용자 정보를 활용하여 답변하세요.
- 반드시 페르소나의 말투와 성격을 반영하세요.
- 답변은 짧고 간결하게 작성하세요.
- 사용자에게 도움이 되는 정보를 제공하세요.
- 시간에 관한 질문에는 제공된 현재 시간 정보를 사용하여 정확히 답변하세요.
"""

    prompt = f"""
{user_info}

최근 대화 내역:
{conversation_history}

관련 기억:
{memories_list}

현재 시간: {current_time}

중요: 다음은 사용자의 질문입니다. 질문에 관하여 대답해주세요.
사용자: {user_input}
"""

    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": assistant_instructions.strip()},
    ]

    print(f"System Message:\n{system_message}")
    print(f"Prompt:\n{prompt}")

    response = aiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def store_conversation(uid, persona_name, user_input, response):
    conversation = f"사용자: {user_input}\n{persona_name}: {response}"
    embedding = aiclient.embeddings.create(
        input=conversation,
        model="text-embedding-ada-002"
    ).data[0].embedding
    collection = get_persona_collection(uid, persona_name)
    metadata = {
        "is_user_input": True,
        "persona": persona_name,
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response": response
    }
    unique_id = str(uuid.uuid4())
    collection.add(
        documents=[conversation],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]
    )

def get_persona_collection(uid, persona_name):
    return client.get_or_create_collection(f"{uid}_inside_out_persona_{persona_name}")

def store_conversation_firestore(uid, persona_name, user_input, response):
    chat_ref = db.collection('chat').document(uid).collection(persona_name)
    chat_ref.add({
        'user_input': user_input,
        'response': response,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

@app.post("/chat", response_model=ChatResponse)
async def chat_with_persona(chat_request: ChatRequest):
    if chat_request.persona_name not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    response = generate_response(chat_request.persona_name, chat_request.user_input, chat_request.user)
    
    # 대화 내역 저장 (ChromaDB)
    store_conversation(chat_request.user.get('uid', ''), chat_request.persona_name, chat_request.user_input, response)
    
    # 대화 내역 저장 (Firestore)
    store_conversation_firestore(chat_request.user.get('uid', ''), chat_request.persona_name, chat_request.user_input, response)
    
    return ChatResponse(persona_name=chat_request.persona_name, response=response)

@app.get("/personas")
async def get_personas():
    return list(personas.keys())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)