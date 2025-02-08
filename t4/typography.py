# t4/typography.py
def pretty_bytes(num):
    # 簡單轉換 bytes 成 KB/MB/GB 字串表示
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"
