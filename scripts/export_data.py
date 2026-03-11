import sqlite3
import json
import random
import os

DB_PATH = "data/teacher.db"
OUTPUT_DIR = "data/finetune"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
EVAL_FILE = os.path.join(OUTPUT_DIR, "eval.json")
EVAL_RATE = 0.1

def export_to_alpaca():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT seed_text, response_json FROM tasks WHERE status = 1")
    rows = cur.fetchall()

    if not rows:
        print("数据库中无处理成功的数据")
        return 
    print(f"发现 {len(rows)} 条有效语料，正在进行格式转换...")

    dataset = []
    for seed, response in rows:
        dataset.append({
            "instruction": "请将下面的自然语言指令解析为 JSON 格式的任务数据。",
            "input": seed,
            "output": response
        })

    random.shuffle(dataset)
    split_index = int(len(dataset) * (1-EVAL_RATE))
    train_data = dataset[:split_index]
    eval_data = dataset[split_index:]

    with open(TRAIN_FILE, mode="w", encoding="utf_8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"导出完成！")
    print(f"训练集: {TRAIN_FILE} ({len(train_data)} 条)")
    print(f"验证集: {EVAL_FILE} ({len(eval_data)} 条)")

if __name__ == "__main__":
    export_to_alpaca()