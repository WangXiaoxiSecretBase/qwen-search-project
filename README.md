# Qwen长思考与搜索工具训练方案

## 项目概述

本项目基于DeepSeek-R1-Distill-Qwen-7B合成训练数据，使用LoRA方法训练Qwen2.5-0.5B-Instruct模型，使其具备长思考能力和判断是否使用网络搜索工具的能力。

## 环境配置

### 推荐环境路径
```bash
/remote-home1/moss/miniconda3/envs/py312
```

## 实验流程

### 1. 数据合成
使用DeepSeek-R1-Distill-Qwen-7B合成训练数据：

```bash
python data_synthesis.py
```

**功能特点：**
- 自动判断问题是否需要搜索工具
- 生成包含长思考过程的回复
- 模拟搜索工具调用和结果处理
- 支持批量处理问题

### 2. 数据格式化
将合成数据转换为Qwen训练格式：

```bash
python data_formatter.py
```

**输出文件：**
- `qwen_data_train.jsonl`: 训练数据（300条）
- `qwen_data_test.jsonl`: 测试数据（85条）

### 3. LoRA训练
基于Qwen2.5-0.5B-Instruct进行LoRA训练：

```bash
python lora_train.py
```

**训练配置：**
- LoRA rank: 16
- LoRA alpha: 32
- 目标模块: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 训练轮数: 3
- 学习率: 2e-4
- 批量大小: 4
- 梯度累积: 4

### 4. 模型测试

```bash
python demo_qwen.py
