import random

VERBS = [
    "帮我记一下：", "新建一个任务：", "把任务栏加上：", "我要做：", "备忘录加上：", "添加任务："
]

OBJECTS = [
    "复习CS231n的CNN笔记",
    "完成taskmaster命令行的SQLite模块开发",
    "完成降低大模型幻觉的最新文献阅读",
    "完成轻量级本地模型的性能优化测试",
    "做Python装饰器的异常处理代码编写",
    "将Git仓库的代码合并",
    "写完Java类型转换的底层逻辑测试",
    "学习多线程汇编模型",
    "拿楼下的外卖",
    "给导师发周报",
    "交水电气费",
    "去健身房练腿"
]

TIMES = [
    "明天早上8点半", "下周三下午", "2026年3月15日", "本月底", 
    "两小时后", "今天下班前", "周末", 
    "抽空", "随便哪天"
]

PRIORITIES = [
    "，十万火急", "，优先级拉满", "，最高优", "，马上要交", "非常紧迫",
    "，正常处理", "，中等优先级", "不太急",
    "，不着急", "，慢慢来", "，顺手做一下"
]


def build_prompt():
    with open("seeds.txt", mode="w", encoding="utf-8") as f:
        for i in range(2000):
            wordV = random.choice(VERBS)
            wordO = random.choice(OBJECTS)
            if random.random() > 0.6:
                wordT = random.choice(TIMES)
            else:
                wordT = ""
            if random.random() > 0.3:
                wordP = random.choice(PRIORITIES)
            else:
                wordP = ""
            f.write(wordV + wordT + wordO + wordP + "\n")
        

