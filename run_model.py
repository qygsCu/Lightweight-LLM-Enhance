# 大模型本地运行：
# 1.将prompt切成词并转换成数字矩阵
# 2.模型前向传播
# 3.数字文本化

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-0.5B"

# from_pretrianed: 根据提供的名字或路径（提供model name则在hugging face中寻找，提供地址则使用本地模型），把一个训练好的模型（或分词器）原封不动地在你的内存里重建出来
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """将以下命令行意图解析为 JSON 格式。
可用字段：`command` (必须为 'add', 'delete', 'update', 'list' 之一), `target_id` (整数), `args` (字符串)。

用户输入："把那个编号是 105 的待办事项删掉吧。"

负面约束：输出内容必须可以直接被 Python 的 json.loads() 解析。绝对不要在首尾使用 ```json 和 ``` 这种 Markdown 代码块标记。只输出花括号包裹的 JSON 本身。"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# apply_chat_template: SFT，将大模型功能从续写转为回答问题。不同模型使用自己特定的格式框定上下文，使LLM回答问题。
# 这个函数的作用是将各个模型的规定模板翻译成模型独有的底层“方言”格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)