채팅 저장 구조
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
