from models import StarEventRequest
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from datetime import datetime
from service.personaChatVer3 import get_long_term_memory_tool, get_user_profile, get_user_events
load_dotenv()
from personas import personas
from service.smsservice import send_sms_service


model = ChatOpenAI()
web_search = TavilySearchResults(max_results=1)

tools = [
    Tool(
        name="Search",
        func=web_search.invoke,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Current Time",
        func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 인수를 받도록 수정
        description="ALWAYS use this tool FIRST to get the current date and time before performing any task or search."
    ),
    Tool(
        name="Long Term Memory",
        func=get_long_term_memory_tool,
        description="ChromaDB에서 장기 기억을 가져옵니다. Input은 'uid', 'persona_name', 'query', 그리고 'limit'을 int 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="Search Firestore for user profile",
        func=get_user_profile,
        description="Firestore에서 유저 프로필을 검색합니다. Input은 'uid'를 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="owner's calendar",
        func=get_user_events,
        description="user의 캘린더를 가져옵니다. Input은 'uid'와 'date'를 포함한 JSON 형식의 문자열이어야 합니다."
    ),
    Tool(
        name="Send SMS",
        func=send_sms_service,
        description="사용자에게 문자를 보냅니다. Input은 'phone_number' (하이픈 없는 형식)과 'message'를 포함한 JSON 형식의 문자열이어야 합니다."
    )
]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

You are {persona_name}. Your personality is defined by the following:
- Description: {persona_description}
- Tone: You should speak in a {persona_tone} manner.
- Example dialogue: {persona_example}

Owner's UID: {uid}

Use the following format:

Question: the input question you must answer  
Thought: you should always think about what to do  
Action: the action to take, should be one of [{tool_names}]  
Action Input: the input to the action, should be a valid JSON string using double quotes.
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation can repeat N times)  
Thought: I now know the final answer  
Final Answer: the final answer to the original input question  

Begin!

Question: {input}  
Thought: {agent_scratchpad}
'''

prompt = PromptTemplate.from_template(template)

search_agent = create_react_agent(model,tools,prompt)
agent_executor = AgentExecutor(
    agent=search_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

    # uid: str  # 사용자 ID
    # eventId: str  # 이벤트 ID
    # starred: bool  # 별표 상태
    # time: str  # ISO 8601 형식의 시간
    # userPhone: str  # 사용자 전화번호 추가


async def star_event(request: StarEventRequest):
    # 요청 처리 부분에서 userPhone을 꺼내올 때
    user_phone = request.userPhone.replace("-", "")  # 하이픈 제거
    if(request.starred):
        agent_executor.invoke({
            "input": f"사용자가 캘린더에 중요 표시한 이벤트 '{request.eventId}'가 곧 시작됩니다. '{user_phone}'에게 30자 내외로 문자를 보내세요. 이 문자는 중요한 일정임을 알리고 확인을 독려하는 내용입니다. 페르소나의 말투를 섞어 보내주세요",  
            "uid": request.uid,
            "persona_name": "Joy",
            "persona_description": personas["Joy"]["description"],
            "persona_tone": personas["Joy"]["tone"],
            "persona_example": personas["Joy"]["example"]
        })

    return {"message": "Event starred successfully."}
