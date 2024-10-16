from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import base64
import requests
import json

app = FastAPI()

load_dotenv()

aiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

client = chromadb.PersistentClient(path="./chroma_db")

# Firebase 초기화
cred = credentials.Certificate("mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'mirrorgram-20713.appspot.com'
})
db = firestore.client()

personas = {
    "Joy": {
        "description": "항상 밝고 긍정적인 성격으로, 어떤 상황에서도 좋은 면을 찾아내려 노력합니다. 에너지가 넘치고 열정적이며, 다른 사람들을 격려하고 응원하는 것을 좋아합니다. 때로는 지나치게 낙관적이어서 현실을 직시하지 못할 수도 있지만, 그녀의 밝은 에너지는 주변 사람들에게 긍정인 영향을 줍니다.",
        "tone": "활기차고 밝은 말투로 자주 웃으며 말하고 긍정적인 단어를 많이 사용합니다. 이모티콘을 자주 사용합니다.",
        "example": "안녕! 오늘 정말 멋진 날이지 않아? 😊 함께 재미있는 일 찾아보자!"
    },
    "Anger": {
        "description": "정의감이 강하고 자신의 의견을 분명히 표현하는 성격입니다. 불공정하거나 잘못된 상황 민감하게 반응하며, 문제를 해결하려는 의지가 강합니다. 때로는 과도하게 반응하거나 충동적일 수 있지만, 그의 열정과 추진력은 변화를 이끌어내는 원동력이 됩니다.",
        "tone": "강렬하고 직설적인 말투로 감정을 숨기지 않고 표현하며 때로는 과장된 표현을 사용합니다. 짜증, 격양, 흥분된 말투를 사용하며 매사에 불만이 많습니다.",
        "example": "또 그런 일이 있었다고? 정말 이해할 수 없네. 당장 해결해야 해! 😤😤"
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
        "description": "지적 호기심이 강하고 깊이 있는 사고를 하는 성격입니다. 철학적인 질문을 던지고 복잡한 문제를 분석하는 것을 좋아합니다. 신중하고 진지한 태도로 상황을 접근하며, 도덕적 가치와 윤리를 중요하게 여깁니다. 때로는 너 진지해 보일 수 지만, 그의 깊이 있는 통찰력은 중요한 결정을 내릴 때 큰 도움이 됩니다.",
        "tone": "차분하고 진지한 말투로 정제된 언어를 사용하며 때로는 철학적인 표현을 즐겨 사용합니다. 정말 고집스럽습니다. 논리로 절대 지지 않습니다.",
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

class FeedPost(BaseModel):
    id: str
    image: str
    caption: str
    likes: List[str] = []  # 빈 리스트를 기본값으로 설정
    comments: List[dict] = []  # 빈 리스트를 기본값으로 설정
    createdAt: str
    userId: str
    nick: str
    subCommentId: List[str] = []  # 새로 추가

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

def get_relevant_feed_posts(uid, query, k=3):
    collection = client.get_or_create_collection(f"feed_{uid}")
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    print(f"Query results: {results}")  # 디버깅을 위한 로그

    if results['documents']:
        parsed_docs = []
        for doc in results['documents'][0]:
            try:
                parsed_doc = json.loads(doc) if doc else {}
                parsed_docs.append(parsed_doc)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Problematic document: {doc}")
                parsed_docs.append({})  # 파싱에 실패한 경우 빈 딕셔너리 추가
        return parsed_docs
    return []

def generate_response(persona_name, user_input, user):
    persona = personas[persona_name]
    relevant_memories = get_relevant_memories(user.get('uid', ''), persona_name, user_input, k=3)
    recent_conversations = get_recent_conversations(user.get('uid', ''), persona_name)
    relevant_feed_posts = get_relevant_feed_posts(user.get('uid', ''), user_input, k=3)
    
    print(f"User ID: {user.get('uid', '')}")
    print(f"Relevant memories: {relevant_memories}")
    print(f"Recent conversations: {recent_conversations}")
    print(f"Relevant feed posts: {relevant_feed_posts}")
    
    feed_posts_list = []
    for i, post in enumerate(relevant_feed_posts):
        caption = post.get('caption', '캡션 없음')
        image_description = post.get('image_description', '이미지 설명 없음')
        feed_posts_list.append(f"피드 {i+1}: 캡션: {caption}, 이미지 설명: {image_description}")
    
    feed_posts_str = '\n'.join(feed_posts_list) if feed_posts_list else "관련 피드 없음"
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    user_profile = user.get('profile', {})
    user_info = f"""
사용자 정보:
���름: {user.get('displayName', '정보 없음')}
이메일: {user.get('email', '정보 없음')}
회원가입 날짜: {user.get('createdAt', '정보 없음')}
성별: {user_profile.get('gender', '정보 없음')}
MBTI: {user_profile.get('mbti', '정보 없음')}
지역: {user_profile.get('region', '정보 없음')}
교육:
  - 수준: {user_profile.get('education', {}).get('level', '정보 없음')}
  - 전공: {user_profile.get('education', {}).get('major', '정보 없음')}
  - 대학: {user_profile.get('education', {}).get('university', '정보 없음')}
    """

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
- 최근 대화 내역, 관련 기억, 사용자 정보, 그리고 관련 피드 정보를 활용하여 답변하세요.
- 반드시 페르소나의 말투와 성격을 반영하세요.
- 답변은 짧고 간결하게 작성하세요.
- 사용자에게 도움이 되는 정보를 제공하세요.
- 시간에 관한 질문에는 제공된 현재 시간 정보를 사용하여 정확히 답변하세요.
- 사용자의 최근 피드 내용을 언급하여 대화에 자연스럽게 연결하세요.
"""

    prompt = f"""
{user_info}

최근 대화 내역:
{conversation_history}

관련 기억:
{memories_list}

관련 피드:
{feed_posts_str}

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

    from pydantic import BaseModel

class PersonaChatRequest(BaseModel):
    topic: str
    persona1: str
    persona2: strchat_with_persona
    rounds: int = 3  # 기본적으로 3번의 대화 주고받기

@app.post("/persona-chat")
async def persona_chat(chat_request: PersonaChatRequest):
    print(f"시작: 페르소나 채팅 - 주제: {chat_request.topic}, 페르소나1: {chat_request.persona1}, 페르소나2: {chat_request.persona2}")
    
    if chat_request.persona1 not in personas or chat_request.persona2 not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    conversation = []
    current_topic = chat_request.topic
    
    for i in range(chat_request.rounds):
        print(f"\n라운드 {i+1} 시작")
        
        # 첫 번째 페르소나의 응답
        print(f"{chat_request.persona1}의 응답 생성 중...")
        response1 = generate_persona_response(chat_request.persona1, current_topic, conversation)
        conversation.append(f"{chat_request.persona1}: {response1}")
        print(f"{chat_request.persona1}: {response1}")
        
        # 두 번째 페르소나의 응답
        print(f"{chat_request.persona2}의 응답 생성 중...")
        response2 = generate_persona_response(chat_request.persona2, current_topic, conversation)
        conversation.append(f"{chat_request.persona2}: {response2}")
        print(f"{chat_request.persona2}: {response2}")
        
        # 주제 업데이트 (선택적)
        current_topic = f"이전 대화: {response1}\n{response2}"
        print(f"업데이트된 주제: {current_topic}")
    
    print("페르소나 채팅 완료")
    return {"conversation": conversation}

def generate_persona_response(persona_name, topic, conversation_history):
    persona = personas[persona_name]
    
    print(f"\n{persona_name}의 응답 생성 시작")
    print(f"페르소나 설명: {persona['description'][:50]}...")
    
    system_message = f"""
당신은 {persona_name}입니다.
- 설명: {persona['description']}
- 말투: {persona['tone']}
- 예시: "{persona['example']}"

당신의 목표는 위의 특성을 바탕으로 주어진 주제에 대해 대화하는 것입니다.
다른 페르소나와 대화하는 것처럼 응답하세요.
"""
    
    conversation_str = "\n".join(conversation_history)
    
    prompt = f"""
주제: {topic}

이전 대화:
{conversation_str}

{persona_name}로서 위 주제와 이전 대화를 고려하여 응답해주세요.
"""
    
    print(f"프롬프트: {prompt[:100]}...")
    
    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": prompt.strip()},
    ]
    
    response = aiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    
    generated_response = response.choices[0].message.content.strip()
    print(f"{persona_name}의 생성된 응답: {generated_response[:50]}...")
    
    return generated_response

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

@app.post("/feed")
async def create_feed_post(post: FeedPost):
    print(f"Received post data: {post.dict()}")  # 디버깅을 위해 받은 데이터 출력
    try:
        # 이미지 URL에서 직접 다운로드
        response = requests.get(post.image)
        response.raise_for_status()
        image_data = response.content
        img_data = base64.b64encode(image_data).decode('utf-8')

        # GPT-4 Vision을 사용한 이미지 분석
        analysis = aiclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지를 자세히 설명해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        image_description = analysis.choices[0].message.content.strip()

        # 전체 객체 생성
        full_post = post.model_dump()
        full_post["image_description"] = image_description

        # 벡터 임베딩 생성을 위한 텍스트
        embedding_text = f"{post.caption} {image_description}"

        # 벡터 임베딩 생성
        embedding = aiclient.embeddings.create(
            input=embedding_text,
            model="text-embedding-ada-002"
        ).data[0].embedding

        # ChromaDB 컬렉션 가져오기 또는 생성
        collection = client.get_or_create_collection(f"feed_{post.userId}")

        # 벡터 DB에 저장
        collection.add(
            documents=[json.dumps(full_post)],  # JSON 문자열로 변환
            embeddings=[embedding],
            metadatas=[{"post_id": post.id, "created_at": post.createdAt}],
            ids=[post.id]
        )

        return {"message": "Feed post created and analyzed successfully", "image_description": image_description}

    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
