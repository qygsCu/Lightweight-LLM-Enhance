import asyncio
from openai import AsyncOpenAI
import sqlite3

API_KEY = "sk-33bf6506a45c452e91ab39248b48a73d"
BASE_URL = "https://api.deepseek.com"
DB_PATH = "data/teacher.db"
SYSTEM_PROMPT = """你是一个精准的结构化数据提取引擎。
请将输入转化为 JSON。输出必须是合法的 JSON 对象。
注意：不要输出 Markdown 代码块，不要有废话。"""
CONCURRENT_LIMIT = 5
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_parser",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "due_date": {"type": ["string", "null"]},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["title", "due_date", "priority"]
        }
    }
}


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    create_db_sql = """CREATE TABLE IF NOT EXISTS tasks(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        seed_text TEXT UNIQUE,
                        response_json TEXT,
                        status INTEGER DEFAULT 0)"""
    cur.execute(create_db_sql)
    cur.execute("SELECT COUNT(*) FROM tasks")
    if cur.fetchone()[0] == 0:
        with open("data/seeds.txt", mode="r", encoding="utf-8") as f:
            seeds = [(line.strip(),) for line in f if line.strip()]
            cur.executemany("INSERT OR IGNORE INTO tasks(seed_text) VALUES(?)", seeds)
    con.commit()
    return con

async def process_task(semaphore, client, task_id, seed_text, conn):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": seed_text}
                ],
                response_format={'type': "json_object"},
                stream=False
            )
            result_raw = response.choices[0].message.content

            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tasks SET response_json = ?, status = 1 WHERE id = ?",
                (result_raw, task_id)
            )
            conn.commit()
            print(f" 成功处理 ID {task_id}")

        except Exception as e:
            print(f"处理 ID {task_id} 错误: {e}")

async def main():
    conn = init_db()
    client = AsyncOpenAI(
        api_key=API_KEY, 
        base_url=BASE_URL
    )
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    while True:
        cursor = conn.cursor()
        cursor.execute("SELECT id, seed_text FROM tasks WHERE status = 0 LIMIT 10")
        rows = cursor.fetchall()
        
        if not rows:
            print("所有任务处理完毕！")
            break
            
        tasks = [process_task(semaphore, client, r[0], r[1], conn) for r in rows]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

