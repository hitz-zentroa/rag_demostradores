# utils/utils.py
import time

def today():
    return time.strftime("%A %B %e, %Y", time.gmtime())

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'