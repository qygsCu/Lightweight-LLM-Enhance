import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAINED_MODEL_PATH = "LLaMA-Factory/models/taskmaster_final"
ORIGINAL_MODEL_PATH = "Qwen/Qwen2.5-0.5B"
TEST_DATA_PATH = "data/finetune/eval.json"

tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_trained = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
model_origin = AutoModelForCausalLM.from_pretrained(
    ORIGINAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_cases = json.load(f)


num_of_correct_trained = 0
num_of_correct_origin = 0

for i, case in enumerate(test_cases):
    instruction = case["instruction"]
    user_input = case["input"]
    expected_output = case["output"]

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(model_trained.device)

    generated_ids_trained = model_trained.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask, 
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.1,                 
        pad_token_id=tokenizer.pad_token_id,    
        eos_token_id=tokenizer.eos_token_id    
    )

    generated_ids_origin = model_origin.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask, 
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.1,                 
        pad_token_id=tokenizer.pad_token_id,    
        eos_token_id=tokenizer.eos_token_id    
    )
    
    # 裁剪掉输入部分，只保留回复
    response_ids_trained = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_trained)
    ]
    response_ids_origin = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_origin)
    ]
    response_trained = tokenizer.batch_decode(response_ids_trained, skip_special_tokens=True)[0]
    response_origin = tokenizer.batch_decode(response_ids_origin, skip_special_tokens=True)[0]
    clean_response_trained = response_trained.split("Human:")[0].split("<|im_end|>")[0].strip()
    clean_response_origin = response_origin.split("Human:")[0].split("<|im_end|>")[0].strip()
    print(f"【测试 {i+1}】")
    print(f"用户输入: {user_input}")
    print(f"预期结果: {expected_output}")
    print(f"训练模型输出: {clean_response_trained}")
    print(f"原模型输出: {clean_response_origin}")
    try:
        pred_dict = json.loads(clean_response_trained)
        if len(pred_dict) == 3 and "title" in pred_dict and "due_date" in pred_dict and "priority" in pred_dict:
            num_of_correct_trained += 1
            print("训练模型正确")
        else:
            print("训练模型json格式正确，但解析错误")
    except:
        print("训练模型json格式错误")
    try:
        pred_dict = json.loads(clean_response_origin)
        if len(pred_dict) == 3 and "title" in pred_dict and "due_date" in pred_dict and "priority" in pred_dict:
            num_of_correct_origin += 1
            print("原模型正确")
        else:
            print("原模型json格式正确，但解析错误")
    except:
        print("原模型错误")
    
    
print(f"训练模型准确率:{num_of_correct_trained / len(test_cases) * 100}%")
print(f"原模型准确率:{num_of_correct_origin / len(test_cases) * 100}%")