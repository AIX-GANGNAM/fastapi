from datetime import datetime
import json
import uuid

def get_current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_unique_id():
    return str(uuid.uuid4())

def parse_firestore_timestamp(timestamp):
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "시간 정보 없음"