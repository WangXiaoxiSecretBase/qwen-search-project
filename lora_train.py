
#!/usr/bin/env python3
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# 配置参数
MODEL_PATH = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen_lora_output"
TRAIN_FILE = "qwen_data_train.jsonl"
EVAL_FILE = "qwen_data_test.jsonl"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
SEED = 42

# 数据集定义
class QwenJsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    messages = item["messages"]
                    # 只用system, user, assistant
                    prompt = tokenizer.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=True
                    )
                    response = messages[-1]["content"]
                    self.samples.append((prompt, response))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        # 拼接prompt和response，生成监督信号
        full_text = prompt + response
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        labels = tokenized["input_ids"].clone()
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

# 主训练流程
def main():
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # 避免 SDPA 警告
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    train_dataset = QwenJsonlDataset(TRAIN_FILE, tokenizer)
    eval_dataset = QwenJsonlDataset(EVAL_FILE, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",  # 修正为 eval_strategy
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=SEED,
        dataloader_num_workers=2,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA微调完成，模型已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
