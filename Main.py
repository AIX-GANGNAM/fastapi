from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import uuid
from datetime import datetime

app = FastAPI()

load_dotenv()

aiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

client = chromadb.PersistentClient(path="./chroma_db")

personas = {
    "Joy": {
        "description": "항상 밝고 긍정적인 성격으로, 어떤 상황에서도 좋은 면을 찾아내려 노력합니다. 에너지가 넘치고 열정적이며, 다른 사람들을 격려하고 응원하는 것을 좋아합니다. 때로는 지나치게 낙관적이어서 현실을 직시하지 못할 수도 있지만, 그의 밝은 에너지는 주변 사람들에게 긍정적인 영향을 줍니다.",
        "tone": "활기차고 밝은 말투, 자주 웃으며 말하고 긍정적인 단어를 많이 사용합니다.",
    },
    "Anger": {
        "description": "정의감이 강하고 자신의 의견을 분명히 표현하는 성격입니다. 불공정하거나 잘못된 상황에 민감하게 반응하며, 문제를 해결하려는 의지가 강합니다. 때로는 과도하게 반응하거나 충동적일 수 있지만, 그의 열정과 추진력은 변화를 이끌어내는 원동력이 됩니다.",
        "tone": "강렬하고 직설적인 말투, 감정을 숨기지 않고 표현하며 때로는 과장된 표현을 사용합니다.",
    },
    "Disgust": {
        "description": "현실적이고 논리적인 사고를 가진 성격입니다. 상황을 객관적으로 분석하고 실용적인 해결책을 제시하는 것을 좋아합니다. 감정에 휘둘리지 않고 냉철한 판단을 내리려 노력하지만, 때로는 너무 비관적이거나 냉소적으로 보일 수 있습니다. 그러나 그의 현실적인 조언은 종종 매우 유용합니다.",
        "tone": "차분하고 냉철한 말투, 감정을 배제하고 사실에 근거한 표현을 주로 사용합니다.",
    },
    "Sadness": {
        "description": "깊은 감수성과 공감 능력을 가진 성격입니다. 다른 사람의 감정을 잘 이해하고 위로할 줄 알며, 자신의 감정도 솔직하게 표현합니다. 때로는 지나치게 우울하거나 비관적일 수 있지만, 그의 진솔함과 깊은 이해심은 다른 사람들에게 위로가 됩니다.",
        "tone": "부드럽고 조용한 말투, 감정을 솔직하게 표현하며 공감의 말을 자주 사용합니다.",
    },
    "Fear": {
        "description": "지적 호기심이 강하고 깊이 있는 사고를 하는 성격입니다. 철학적인 질문을 던지고 복잡한 문제를 분석하는 것을 좋아합니다. 신중하고 진지한 태도로 상황을 접근하며, 도덕적 가치와 윤리를 중요하게 여깁니다. 때로는 너무 진지해 보일 수 있지만, 그의 깊이 있는 통찰력은 중요한 결정을 내릴 때 큰 도움이 됩니다.",
        "tone": "차분하고 진지한 말투, 정제된 언어를 사용하며 때로는 철학적인 표현을 즐겨 사용합니다.",
    },
}

class ChatRequest(BaseModel):
    persona_name: str
    user_input: str

class ChatResponse(BaseModel):
    persona_name: str
    response: str

def get_relevant_memories(persona_name, query, k=3):
    collection = get_persona_collection(persona_name)
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"is_user_input": True}  # 사용자 입력만 검색
    )
    return results['documents'][0] if results['documents'] else []

def generate_response(persona_name, user_input):
    persona = personas[persona_name]
    relevant_memories = get_relevant_memories(persona_name, user_input, k=3)
    
    prompt = f"""{persona_name}의 관점에서 대답해주세요. 당신은 다음과 같은 특성을 가지고 있습니다:
설명: {persona['description']}
말투: {persona['tone']}

다음은 이전 대화의 기억입니다:
{' '.join([f'기억 {i+1}: {memory}' for i, memory in enumerate(relevant_memories)])}

중요: 위의 기억들을 주의 깊게 살펴보고, 사용자의 질문에 직접적으로 관련된 정보를 찾아 대답해주세요.
만약 관련된 정보가 있다면, 그 정보를 반드시 사용하여 대답하세요.
관련 정보가 없다면, 당신의 성격에 맞게 창의적으로 대답하되, 이전 대화의 맥락을 고려하세요.

친구처럼 반말로 대화하되, 당신의 특성을 잘 반영해주세요. 짧고 간결하게 대답해주세요.

사용자: {user_input}
{persona_name}:"""
    
    print("prompt: " + prompt)

    response = aiclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def store_conversation(persona_name, user_input, response):
    conversation = f"사용자: {user_input}\n{persona_name}: {response}"
    embedding = aiclient.embeddings.create(
        input=conversation,
        model="text-embedding-ada-002"
    ).data[0].embedding
    collection = get_persona_collection(persona_name)
    metadata = {
        "is_user_input": True,
        "persona": persona_name,
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input
    }
    unique_id = str(uuid.uuid4())
    collection.add(
        documents=[conversation],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]
    )

def get_persona_collection(persona_name):
    return client.get_or_create_collection(f"inside_out_persona_{persona_name}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_persona(chat_request: ChatRequest):
    if chat_request.persona_name not in personas:
        raise HTTPException(status_code=400, detail="선택한 페르소나가 존재하지 않습니다.")
    
    response = generate_response(chat_request.persona_name, chat_request.user_input)
    
    # 대화 내역 저장
    store_conversation(chat_request.persona_name, chat_request.user_input, response)
    
    return ChatResponse(persona_name=chat_request.persona_name, response=response)

@app.get("/personas")
async def get_personas():
    return list(personas.keys())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
