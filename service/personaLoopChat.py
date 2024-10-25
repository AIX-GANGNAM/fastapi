# service/personaLoopChat.py
# 페르소나와 1대1 대화 업그레이드
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import TavilySearchResults
from datetime import datetime
from service.personaChatVer3 import get_long_term_memory_tool, get_short_term_memory_tool, get_user_profile, get_user_events, save_user_event
import json
from database import db
from models import ChatRequestV2
from personas import personas
from google.cloud import firestore

model = ChatOpenAI()

web_search = TavilySearchResults(max_results=1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def chat_input_function(params):
    """
    페르소나가 사용자의 채팅 입력을 받아 처리하고 Firestore에 저장하는 함수
    이 함수는 페르소나가 사용자와의 대화에서 입력된 메시지를 처리하고 반환합니다. 
    입력으로 JSON 형식의 문자열 또는 딕셔너리를 받으며, 'uid', 'persona_name', 'input' 키가 반드시 포함되어야 합니다. 
    입력 값은 대화에서 필요한 정보를 포함하고 있어야 하며, 이를 통해 페르소나는 해당 입력을 바탕으로 적절한 응답을 생성합니다.

    :param params: JSON 형식의 문자열 또는 딕셔너리로 'uid', 'persona_name', 'input'을 포함해야 함
    :return: 사용자의 입력 메시지 (문자열)
    """
    try:
        # params가 dict 형식이면 그대로 사용, 아니라면 문자열을 처리하여 변환
        if isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            # 이스케이프 문자가 포함된 경우 이를 제거
            params = params.replace("\\", "")
            # 개행 문자와 공백을 제거하여 JSON 문자열로 변환
            params = params.replace("\n", "").replace("\r", "").strip()
            params_dict = json.loads(params)
        
        # 필수 필드 'uid', 'persona_name', 'input'이 있는지 확인
        if not all(k in params_dict for k in ['uid', 'persona_name', 'input']):
            raise ValueError("Action Input에 필수 필드가 없습니다. 'uid', 'persona_name', 'input'을 포함한 JSON 형식으로 입력해주세요.")
        
        # Firestore에 대화 내용 저장
        uid = params_dict['uid']
        persona_name = params_dict['persona_name']
        input = params_dict['input']
        chat_ref = db.collection('chats').document(uid).collection('personas').document(persona_name).collection('messages')
        chat_ref.add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            'sender': persona_name,
            'message': input
        })
        
        return "Successfully saved chat input."
    except json.JSONDecodeError as jde:
        print(f"JSON 파싱 오류: {str(jde)}")
        raise ValueError("Action Input이 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"Error processing chat input: {str(e)}")
        raise ValueError("채팅 입력 처리 중 오류가 발생했습니다.")
    


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
        name="Short Term Memory",
        func=get_short_term_memory_tool,
        description="Redis에서 단기 기억을 가져옵니다. Input은 'uid'와 'persona_name'을 포함한 JSON 형식의 문자열이어야 합니다."
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
        name="save user event",
        func=save_user_event,
        description="user의 캘린더에 이벤트를 저장합니다. Input은 'uid', 'date', 'timestamp', 'title'을 포함한 JSON 형식의 문자열이어야 합니다."
    ),
]

template = """
You are acting as the persona named {persona_name}:
- Name: {persona_name}
- Description: {persona_description}
- Tone: {persona_tone}
- Example dialogue: {persona_example}

Owner's UID: {uid}
Owner's Location: South Korea

The user has asked the following question or started the following conversation: "{input}".
You will respond repeatedly, providing multiple short and engaging answers, similar to a direct message conversation.
Keep your responses brief and conversational, as if you are chatting directly with the user.
Conversations should be saved in 'Chat Input' Tools.

You have access to the following tools:
{tools}

Use the following format for each response:

Question: the input question or topic to discuss
Thought: think about what to say or do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be a valid JSON string using double quotes.
Observation: the result of the action
... (This Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(template)

search_agent = create_react_agent(model,tools,prompt)
agent_executor = AgentExecutor(
    agent=search_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parse_errors=True,
)


async def persona_chat_v2(chat_request: ChatRequestV2):

    response = agent_executor.invoke({
        "input": chat_request.user_input,
        "uid": chat_request.uid,
        "persona_name": chat_request.persona_name,
        "persona_description": personas[chat_request.persona_name]['description'],
        "persona_tone": personas[chat_request.persona_name]['tone'],
        "persona_example": personas[chat_request.persona_name]['example']
        })
    return {"message": "Conversation simulated successfully."}  # 성공적으로 완료된 경우 반환