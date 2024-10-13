# Inside Out Persona Chat API

이 프로젝트는 Disney/Pixar의 "Inside Out" 영화에서 영감을 받은 감정 페르소나 채팅 API를 구현한 FastAPI 애플리케이션입니다. 사용자는 다양한 감정 페르소나(Joy, Anger, Disgust, Sadness, Fear)와 대화할 수 있으며, 각 페르소나는 고유한 성격과 말투를 가지고 있습니다.

## 주요 기능

1. 다중 페르소나 채팅: 5개의 감정 페르소나와 대화 가능
2. 맥락 인식 응답: 이전 대화 내역과 사용자 정보를 고려한 응답 생성
3. 벡터 데이터베이스 기반 관련 기억 검색
4. Firebase Firestore를 이용한 대화 내역 저장 및 조회
5. OpenAI의 GPT 모델을 활용한 자연스러운 대화 생성

### 채팅 저장 구조

```
chat (collection)
  └─ uid (document)
     └─ persona_name (collection)
        ├─ auto_id_1 (document)
        │  ├─ user_input: "사용자 메시지"
        │  ├─ response: "GPT 응답"
        │  └─ timestamp: 서버 타임스탬프
        ├─ auto_id_2 (document)
        │  ├─ user_input: "다음 사용자 메시지"
        │  ├─ response: "다음 GPT 응답"
        │  └─ timestamp: 서버 타임스탬프
        └─ ...
```

실시간 업데이트: Firestore의 실시간 리스너를 사용하여 새 메시지가 추가될 때 프론트엔드에서 즉시 반영할 수 있습니다.

채팅 불러오기
```javascript
import { getFirestore, collection, query, orderBy, limit, onSnapshot } from "firebase/firestore";

const db = getFirestore();

function loadChatHistory(uid, personaName, limitCount = 50) {
  const chatRef = collection(db, "chat", uid, personaName);
  const q = query(chatRef, orderBy("timestamp", "desc"), limit(limitCount));

  onSnapshot(q, (snapshot) => {
    const messages = [];
    snapshot.forEach((doc) => {
      messages.push({ id: doc.id, ...doc.data() });
    });
    // 시간 순으로 정렬 (오래된 메시지부터)
    messages.reverse();
    // 여기서 messages를 사용하여 UI를 업데이트합니다
    updateChatUI(messages);
  });
}

// 사용 예:
loadChatHistory("user123", "Joy");
```

## 기술 스택

- FastAPI: 웹 API 프레임워크
- OpenAI API: 자연어 처리 및 대화 생성
- ChromaDB: 벡터 데이터베이스 (관련 기억 저장 및 검색)
- Firebase Admin SDK: Firestore 데이터베이스 연동
- Pydantic: 데이터 검증 및 설정 관리
- python-dotenv: 환경 변수 관리

## 설치 및 설정

1. 저장소 클론:
   ```
   git clone [저장소 URL]
   cd [프로젝트 디렉토리]
   ```

2. 가상 환경 생성 및 활성화:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치:
   ```
   pip install -r requirements.txt
   ```

4. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 내용을 추가:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Firebase 서비스 계정 키 설정:
   Firebase 콘솔에서 서비스 계정 키(JSON)를 다운로드하고 프로젝트 루트에 저장

6. `Main.py` 파일에서 Firebase 초기화 부분 수정:
   ```python
   cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
   ```

## 실행 방법

```
uvicorn Main:app --reload
```

서버가 시작되면 `http://localhost:8000`에서 API에 접근할 수 있습니다.

## API 엔드포인트

### POST /chat

페르소나와 대화를 시작합니다.

요청 본문:
```json
{
  "persona_name": "Joy",
  "user_input": "안녕하세요!",
  "user": {
    "uid": "user123",
    "profile": {
      "userName": "홍길동",
      "birthday": "1990-01-01",
      "mbti": "ENFP",
      "personality": "활발하고 긍정적인"
    }
  }
}
```

응답:
```json
{
  "persona_name": "Joy",
  "response": "안녕, 홍길동! 오늘도 너의 활발하고 긍정적인 에너지가 느껴져서 너무 좋아! 우리 함께 멋진 하루를 만들어보자!"
}
```

### GET /personas

사용 가능한 페르소나 목록을 반환합니다.

응답:
```json
["Joy", "Anger", "Disgust", "Sadness", "Fear"]
```

## 주요 컴포넌트 설명

### 페르소나 정의 (personas 딕셔너리)

각 페르소나의 성격과 말투를 정의합니다.

### generate_response 함수

사용자 입력에 대한 페르소나의 응답을 생성합니다. 다음 요소를 고려합니다:
- 페르소나의 특성
- 사용자 정보
- 최근 대화 내역
- 관련 기억

### get_relevant_memories 함수

ChromaDB를 사용하여 현재 대화와 관련된 이전 대화를 검색합니다.

### get_recent_conversations 함수

Firestore에서 최근 대화 내역을 가져옵니다.

### store_conversation 및 store_conversation_firestore 함수

대화 내용을 ChromaDB와 Firestore에 각각 저장합니다.

## 확장 및 개선 방안

1. 더 많은 페르소나 추가
2. 사용자 인증 및 권한 관리 구현
3. 대화 내역 분석 및 인사이트 제공 기능
4. 다국어 지원
5. 웹 인터페이스 또는 모바일 앱 개발
