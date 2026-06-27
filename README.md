# Lightweight-LLM-Enhance

一个面向中文任务理解的小型实战项目：
通过 "教师模型 API 蒸馏 -> 构造指令数据集 -> LoRA 微调 Qwen2.5-0.5B -> 导出本地模型 -> 对比评测"，验证轻量模型在本地部署场景中的效果提升。

当前任务场景是将自然语言待办指令解析为结构化 JSON：

```json
{"title": "去健身房练腿", "due_date": "2026-04-08 00:00", "priority": "high"}
```

## 项目亮点

- 自动生成种子语料（任务描述、时间、优先级组合）
- 使用教师模型批量标注，落库到 SQLite
- 自动导出 Alpaca 格式训练/验证集
- 基于 LLaMA-Factory 进行 Qwen LoRA SFT
- 对比 "原模型 vs 微调模型" 的结构化输出准确率

## 目录结构

```text
.
├── data/
│   ├── seeds.txt
│   └── finetune/
│       ├── train.json
│       └── eval.json
├── prompts/
│   └── teacher_v1.txt
├── scripts/
│   ├── generate_seeds.py    # 生成种子语料
│   ├── distill_api.py       # 教师模型标注并写入 SQLite
│   ├── export_data.py       # 导出 train/eval
│   ├── inference_test.py    # 原模型/微调模型对比评测
│   └── run_model.py         # 本地模型推理示例
└── LLaMA-Factory/
    ├── data/dataset_info.json
    ├── saves/qwen_taskmaster_lora/
    └── models/taskmaster_final/
```

## 环境准备

推荐 Python `3.10+`。

### 1) 安装根目录依赖

```bash
pip install -r requirements.txt
```

### 2) 安装 LLaMA-Factory

```bash
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
cd ..
```

## 数据构建流程

请在项目根目录执行以下命令。

### 1) 生成种子文本

```bash
python scripts/generate_seeds.py
```

输出：`data/seeds.txt`（默认约 2000 条）。

### 2) 教师模型蒸馏标注

```bash
python scripts/distill_api.py
```

输出：`data/teacher.db`，表 `tasks` 内含：
- `seed_text`：原始指令
- `response_json`：教师模型结构化结果
- `status`：处理状态

注意：`scripts/distill_api.py` 当前使用脚本内 API Key，建议改为环境变量管理后再使用。

### 3) 导出训练/验证集

```bash
python scripts/export_data.py
```

输出：
- `data/finetune/train.json`
- `data/finetune/eval.json`

数据格式为 Alpaca 三字段：`instruction`、`input`、`output`。

## 模型训练（LoRA SFT）

本项目已经在 `LLaMA-Factory/data/dataset_info.json` 注册数据集：`taskmaster_v1`。

可直接使用以下命令复现训练（参数与当前仓库训练记录一致）：

```bash
cd LLaMA-Factory
llamafactory-cli train \
	--model_name_or_path Qwen/Qwen2.5-0.5B \
	--trust_remote_code \
	--stage sft \
	--do_train \
	--finetuning_type lora \
	--dataset taskmaster_v1 \
	--dataset_dir data \
	--template default \
	--cutoff_len 512 \
	--max_samples 100000 \
	--preprocessing_num_workers 16 \
	--output_dir saves/qwen_taskmaster_lora \
	--logging_steps 5 \
	--save_steps 100 \
	--plot_loss \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--learning_rate 5e-5 \
	--num_train_epochs 5 \
	--lr_scheduler_type cosine \
	--lora_rank 8 \
	--lora_alpha 16 \
	--lora_target all \
	--bf16 \
	--report_to none
cd ..
```

训练输出目录：`LLaMA-Factory/saves/qwen_taskmaster_lora`。

## 导出合并模型

将 LoRA 适配器合并为可直接加载的本地模型：

```bash
cd LLaMA-Factory
llamafactory-cli export \
	--model_name_or_path Qwen/Qwen2.5-0.5B \
	--adapter_name_or_path saves/qwen_taskmaster_lora \
	--finetuning_type lora \
	--template default \
	--export_dir models/taskmaster_final
cd ..
```

导出后可在 `LLaMA-Factory/models/taskmaster_final` 看到 `model.safetensors` 等文件。

## 推理与评测

### 1) 单次推理示例

```bash
python scripts/run_model.py
```

### 2) 原模型与微调模型对比

```bash
python scripts/inference_test.py
```

该脚本会读取 `data/finetune/eval.json`，并分别使用：
- 原模型：`Qwen/Qwen2.5-0.5B`
- 微调模型：`LLaMA-Factory/models/taskmaster_final`

按 JSON 字段完整性（`title`/`due_date`/`priority`）统计准确率。

## 常见问题

### 1) 显存不足

- 降低 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 缩短 `cutoff_len`

### 2) 下载模型慢

- 提前在本地缓存 Hugging Face 模型
- 或将 `--model_name_or_path` 改为本地模型路径

### 3) 数据未导出

- 确认 `data/teacher.db` 存在且 `tasks.status=1` 有记录

## 致谢

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen](https://huggingface.co/Qwen)
