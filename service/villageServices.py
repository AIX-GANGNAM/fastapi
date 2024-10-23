# services/agent_service.py
import random
import heapq
import math
from datetime import datetime
from typing import List
from fastapi import WebSocket
from database import redis_client
import json


# 맵 사이즈 설정
map_width = 2000
map_height = 2000

# 노드 정의 (좌표를 프론트엔드와 일치시키기 위해 비율로 설정)
nodes = {
    'home_Joy': {'x': 0.2, 'y': 0.3},
    'home_Sadness': {'x': 0.4, 'y': 0.7},
    'home_Anger': {'x': 0.6, 'y': 0.4},
    'home_Fear': {'x': 0.8, 'y': 0.6},
    'home_Disgust': {'x': 0.2, 'y': 0.7},
    'park': {'x': 0.1, 'y': 0.1},
    'cafe': {'x': 0.8, 'y': 0.7},
    'school': {'x': 0.5, 'y': 0.2},
    'mall': {'x': 0.2, 'y': 0.8},
    # 주요 교차로
    'intersection_1': {'x': 0.3, 'y': 0.5},
    'intersection_2': {'x': 0.7, 'y': 0.5},
}

# 엣지 정의 (인접 리스트)
graph = {
    'home_Joy': ['intersection_1'],
    'home_Sadness': ['intersection_1'],
    'home_Disgust': ['intersection_1'],
    'home_Anger': ['intersection_2'],
    'home_Fear': ['intersection_2'],
    'intersection_1': ['home_Joy', 'home_Sadness', 'home_Disgust', 'intersection_2', 'mall', 'park'],
    'intersection_2': ['home_Anger', 'home_Fear', 'intersection_1', 'school', 'cafe'],
    'park': ['intersection_1'],
    'cafe': ['intersection_2'],
    'school': ['intersection_2'],
    'mall': ['intersection_1'],
}

# A* 알고리즘 구현
def heuristic(a, b):
    return math.hypot(nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y'])

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, 0, start))  # (우선순위, 카운터, 노드)
    came_from = {}
    g_score = {node: float('inf') for node in nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in nodes}
    f_score[start] = heuristic(start, goal)
    counter = 0  # 카운터 초기화

    while open_set:
        current = heapq.heappop(open_set)[2]

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path[1:]  # 시작 노드는 제외하고 반환

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
    return None  # 경로가 없을 경우

# 에이전트 클래스 정의
class Agent:
    def __init__(self, name, home_node):
        self.name = name
        self.home_node = home_node
        # 상태 로드를 시도
        saved_state = redis_client.get(f'agent_state:{name}')
        if saved_state:
            self.load_state(json.loads(saved_state))
        else:
            self.current_node = home_node
            # 위치를 비율로 저장하고, 필요할 때 맵 크기와 곱하여 실제 위치를 계산합니다.
            self.position_ratio = {'x': nodes[home_node]['x'], 'y': nodes[home_node]['y']}
            self.position = {'x': self.position_ratio['x'] * map_width, 'y': self.position_ratio['y'] * map_height}
            self.destination_node = None
            self.path = []
            self.speed = 0.005  # 이동 속도를 비율로 설정
            self.state = 'idle'
            self.following_agent = None  # 추적 중인 에이전트

    def save_state(self):
        state = {
            'name': self.name,
            'home_node': self.home_node,
            'current_node': self.current_node,
            'position_ratio': self.position_ratio,
            'position': self.position,
            'destination_node': self.destination_node,
            'path': self.path,
            'speed': self.speed,
            'state': self.state,
            'following_agent_name': self.following_agent.name if self.following_agent else None,
        }
        redis_client.set(f'agent_state:{self.name}', json.dumps(state))

    def load_state(self, state):
        self.current_node = state.get('current_node', self.home_node)
        self.position_ratio = state.get('position_ratio')
        if not self.position_ratio:
            # position_ratio가 없으면 position을 사용하여 계산
            position = state.get('position')
            if position:
                self.position_ratio = {
                    'x': position['x'] / map_width,
                    'y': position['y'] / map_height
                }
            else:
                # position도 없으면 기본값 설정
                self.position_ratio = {
                    'x': nodes[self.home_node]['x'],
                    'y': nodes[self.home_node]['y']
                }
        self.position = state.get('position')
        if not self.position:
            # position이 없으면 position_ratio를 사용하여 계산
            self.position = {
                'x': self.position_ratio['x'] * map_width,
                'y': self.position_ratio['y'] * map_height
            }
        self.destination_node = state.get('destination_node', None)
        self.path = state.get('path', [])
        self.speed = state.get('speed', 0.005)
        self.state = state.get('state', 'idle')
        # following_agent는 이름으로 매핑
        following_agent_name = state.get('following_agent_name')
        if following_agent_name:
            self.following_agent = agent_manager_instance.agents_dict.get(following_agent_name)
        else:
            self.following_agent = None

    def set_destination(self, destination_node_or_agent):
        if isinstance(destination_node_or_agent, Agent):
            # 다른 에이전트를 따라가는 경우
            self.following_agent = destination_node_or_agent
            self.destination_node = None
            self.path = []
            self.state = 'following'
            print(f"{self.name}이(가) {self.following_agent.name}을(를) 따라갑니다.")
        else:
            # 기존 장소로 이동하는 로직
            self.following_agent = None
            if self.current_node == destination_node_or_agent:
                print(f"{self.name}은 이미 {destination_node_or_agent}에 있습니다.")
                self.state = 'idle'
                self.destination_node = None
                self.path = []
            else:
                self.destination_node = destination_node_or_agent
                self.path = astar(self.current_node, self.destination_node)
                if self.path:
                    self.state = 'moving'
                    print(f"{self.name}이(가) {self.destination_node}로 이동을 시작합니다.")
                else:
                    print(f"{self.name}의 경로를 찾을 수 없습니다.")

    def update_position(self):
        if self.state == 'moving' and self.path:
            next_node = self.path[0]
            target_pos_ratio = nodes[next_node]
            direction_x = target_pos_ratio['x'] - self.position_ratio['x']
            direction_y = target_pos_ratio['y'] - self.position_ratio['y']
            distance = math.hypot(direction_x, direction_y)
            if distance < self.speed:
                # 다음 노드에 도착
                self.position_ratio['x'] = target_pos_ratio['x']
                self.position_ratio['y'] = target_pos_ratio['y']
                self.position['x'] = self.position_ratio['x'] * map_width
                self.position['y'] = self.position_ratio['y'] * map_height
                self.current_node = next_node
                self.path.pop(0)
                if not self.path:
                    self.state = 'idle'
                    self.destination_node = None
                    print(f"{self.name}이(가) {self.current_node}에 도착했습니다.")
            else:
                # 방향 벡터 정규화하여 이동
                self.position_ratio['x'] += (direction_x / distance) * self.speed
                self.position_ratio['y'] += (direction_y / distance) * self.speed
                self.position['x'] = self.position_ratio['x'] * map_width
                self.position['y'] = self.position_ratio['y'] * map_height
        elif self.state == 'following' and self.following_agent:
            # 추적 중인 에이전트의 현재 노드로 경로 재계산
            target_node = self.following_agent.current_node
            if self.current_node != target_node:
                self.path = astar(self.current_node, target_node)
                if self.path:
                    next_node = self.path[0]
                    target_pos_ratio = nodes[next_node]
                    direction_x = target_pos_ratio['x'] - self.position_ratio['x']
                    direction_y = target_pos_ratio['y'] - self.position_ratio['y']
                    distance = math.hypot(direction_x, direction_y)
                    if distance < self.speed:
                        # 다음 노드에 도착
                        self.position_ratio['x'] = target_pos_ratio['x']
                        self.position_ratio['y'] = target_pos_ratio['y']
                        self.position['x'] = self.position_ratio['x'] * map_width
                        self.position['y'] = self.position_ratio['y'] * map_height
                        self.current_node = next_node
                        self.path.pop(0)
                    else:
                        # 방향 벡터 정규화하여 이동
                        self.position_ratio['x'] += (direction_x / distance) * self.speed
                        self.position_ratio['y'] += (direction_y / distance) * self.speed
                        self.position['x'] = self.position_ratio['x'] * map_width
                        self.position['y'] = self.position_ratio['y'] * map_height
                else:
                    print(f"{self.name}이(가) {self.following_agent.name}에게 갈 수 있는 경로를 찾을 수 없습니다.")
            else:
                # 같은 노드에 도착한 경우
                print(f"{self.name}이(가) {self.following_agent.name}과(와) 만났습니다.")
                self.state = 'interacting'
                # 상호작용 로직 호출
                interact(self, self.following_agent)
        elif self.state == 'idle':
            pass  # 현재는 아무 동작도 하지 않음
        elif self.state == 'interacting':
            # 상호작용 중인 경우 처리
            pass
        # 상태 저장
        self.save_state()
    
    def random_move(self):
        # 현재 노드에서 이동할 수 있는 이웃 노드 중 하나를 랜덤 선택
        if self.current_node in graph:
            possible_moves = graph[self.current_node]
            if possible_moves:
                next_node = random.choice(possible_moves)
                self.set_destination(next_node)
        

# 에이전트 매니저 클래스
class AgentManager:
    def __init__(self):
        self.agents = [
            Agent('Joy', 'home_Joy'),
            Agent('Sadness', 'home_Sadness'),
            Agent('Anger', 'home_Anger'),
            Agent('Fear', 'home_Fear'),
            Agent('Disgust', 'home_Disgust'),
        ]
        self.agents_dict = {agent.name: agent for agent in self.agents}
        self.active_connections: List[WebSocket] = []
        # 스케줄 관련 초기화
        self.all_schedules = self.generate_schedules()

    def generate_schedules(self):
        # 스케줄 생성 로직 (현재 시간으로 설정)
        current_time = datetime.now().strftime("%H:%M")
        all_schedules = AllPersonasSchedule(schedules=[
            PersonaSchedule(persona='Joy', schedule=[
                ScheduleItem(time=current_time, interaction_target='intersection_1', topic='대화하기', conversation_rounds=3),
                # 다른 스케줄 항목...
            ]),
            PersonaSchedule(persona='Anger', schedule=[
                ScheduleItem(time=current_time, interaction_target='home_Sadness', topic='산책하기', conversation_rounds=2),
                # 다른 스케줄 항목...
            ]),
            # 다른 페르소나들의 스케줄 추가...
        ])
        return all_schedules

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("클라이언트가 연결되었습니다.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("클라이언트가 연결 해제되었습니다.")

    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"연결에 문제가 발생하여 제거합니다: {e}")
                self.disconnect(connection)

    def apply_schedule_to_agents(self, uid):
        current_time = datetime.now().strftime("%H:%M")
        for persona_schedule in self.all_schedules.schedules:
            for item in persona_schedule.schedule:
                if item.time == current_time and not item.applied:
                    agent = self.agents_dict.get(persona_schedule.persona)
                    if agent:
                        if agent.state == 'idle':
                            # 상호작용 대상이 다른 에이전트인 경우
                            if item.interaction_target in self.agents_dict:
                                other_agent = self.agents_dict[item.interaction_target]
                                self.meet_other_agent(agent, other_agent)
                            else:
                                # 대상이 장소인 경우
                                agent.set_destination(item.interaction_target)
                        item.applied = True  # 스케줄을 적용했음을 표시

    def meet_other_agent(self, agent, other_agent):
        # 상대 에이전트를 따라감
        agent.set_destination(other_agent)

    def update_agents(self):
        for agent in self.agents:
            if agent.state == 'idle':
                agent.random_move()
            agent.update_position()
            print(f"{agent.name}의 현재 상태: {agent.state}, 위치: {agent.position}")

    def get_agents_positions(self):
        return {agent.name: {'x': agent.position['x'], 'y': agent.position['y']} for agent in self.agents}

# 스케줄 클래스 정의
class ScheduleItem:
    def __init__(self, time, interaction_target, topic, conversation_rounds):
        self.time = time
        self.interaction_target = interaction_target
        self.topic = topic
        self.conversation_rounds = conversation_rounds
        self.applied = False  # 스케줄 적용 여부를 추적

class PersonaSchedule:
    def __init__(self, persona, schedule):
        self.persona = persona
        self.schedule = schedule

class AllPersonasSchedule:
    def __init__(self, schedules):
        self.schedules = schedules

def interact(agent1, agent2):
    # 상호작용 로직 구현
    print(f"{agent1.name}이(가) {agent2.name}과(와) 상호작용을 시작합니다.")
    # 상호작용 후 상태를 idle로 변경
    agent1.state = 'idle'
    agent2.state = 'idle'
    agent1.following_agent = None
    agent2.following_agent = None

def get_color_for_agent(name):
    colors = {
        'Joy': '#FFFF00',
        'Anger': '#FF0000',
        'Disgust': '#00FF00',
        'Sadness': '#0000FF',
        'Fear': '#800080'
    }
    return colors.get(name, '#FFFFFF')

def get_traits_for_agent(name):
    traits = {
        'Joy': ['긍정적', '활발함'],
        'Anger': ['열정적', '정의로움'],
        'Disgust': ['깔끔함', '예민함'],
        'Sadness': ['공감능력', '신중함'],
        'Fear': ['조심성', '상상력']
    }
    return traits.get(name, [])

def get_memories_for_agent(name):
    memories = {
        'Joy': ['즐거운 파티', '친구와의 웃음', '성공한 프로젝트'],
        'Anger': ['부당한 대우', '극복한 어려움', '승리의 순간'],
        'Disgust': ['더러운 환경', '불쾌한 냄새', '개선된 상황'],
        'Sadness': ['이별의 순간', '그리운 추억', '우울한 날씨'],
        'Fear': ['무서운 영화', '어두운 밤길', '새로운 도전']
    }
    return memories.get(name, [])

# 싱글톤 인스턴스 저장을 위한 변수
agent_manager_instance = None

def get_agent_manager():
    global agent_manager_instance
    if agent_manager_instance is None:
        agent_manager_instance = AgentManager()
    return agent_manager_instance

def get_all_agents(uid):
    # 에이전트들의 위치 정보를 반환하는 함수
    agent_manager = get_agent_manager()
    agents_data = []
    for index, agent in enumerate(agent_manager.agents):
        agent_data = {
            'id': index,
            'name': agent.name,
            'position': agent.position,
            'color': get_color_for_agent(agent.name),
            'traits': get_traits_for_agent(agent.name),
            'memories': get_memories_for_agent(agent.name),
            'currentEmotion': agent.name.lower(),
            'homeX': agent.position['x'],
            'homeY': agent.position['y']
        }
        agents_data.append(agent_data)
    return agents_data