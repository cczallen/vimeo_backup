# t4/title_to_id.py
import re
def safe_filename(name):
    # 移除不安全的字元
    return re.sub(r'[\\/:"*?<>|]+', "", name)
