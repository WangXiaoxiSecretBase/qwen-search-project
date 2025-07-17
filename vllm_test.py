#!/usr/bin/env python3
"""
使用vLLM进行批量测试
"""

import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os

class VLLMTester:
    def __init__(self, model_path, adapter_path=None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 初始化vLLM
        if adapter_path:
            # 修复：使用正确的 LoRA 配置方式
            self.llm = LLM(
                model=model_path,
                enable_lora=True,
                max_lora_rank=64,  # 设置 LoRA rank
                tensor_parallel_size=1,
                dtype="bfloat16",
                trust_remote_code=True
            )
        else:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="bfloat16",
                trust_remote_code=True
            )
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

    def load_test_data(self, test_file):
        """加载测试数据"""
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line.strip()))
        return test_data

    def prepare_prompts(self, test_data):
        """准备测试提示"""
        prompts = []
        for item in test_data:
            messages = item["messages"]
            # 只使用system和user的消息
            input_messages = [msg for msg in messages if msg["role"] in ["system", "user"]]
            
            prompt = self.tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        return prompts

    def batch_generate(self, prompts):
        """批量生成"""
        print(f"开始批量生成，共 {len(prompts)} 个提示")
        
        # 如果使用 LoRA，需要在生成时指定 LoRA 适配器
        if self.adapter_path:
            # 方法1: 使用 LoRA 请求
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("default", 1, self.adapter_path)
            
            outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                lora_request=lora_request,
                use_tqdm=True
            )
        else:
            # 使用vLLM进行批量推理
            outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                use_tqdm=True
            )
        
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        
        return responses

    def evaluate_responses(self, test_data, responses):
        """评估响应"""
        results = []
        correct_tool_calls = 0
        total_with_search = 0
        total_without_search = 0
        
        for i, (item, response) in enumerate(zip(test_data, responses)):
            expected_search = item.get("use_search", False)
            
            # 检查是否包含工具调用
            has_tool_call = "search" in response and "{" in response
            
            result = {
                "question": item["messages"][1]["content"],  # user message
                "expected_search": expected_search,
                "has_tool_call": has_tool_call,
                "response": response,
                "correct": has_tool_call == expected_search
            }
            
            results.append(result)
            
            if expected_search:
                total_with_search += 1
                if has_tool_call:
                    correct_tool_calls += 1
            else:
                total_without_search += 1
                if not has_tool_call:
                    correct_tool_calls += 1
        
        # 计算准确率
        total_correct = sum(1 for r in results if r["correct"])
        overall_accuracy = total_correct / len(results) if results else 0
        
        print(f"\n=== 评估结果 ===")
        print(f"总体准确率: {overall_accuracy:.3f} ({total_correct}/{len(results)})")
        print(f"需要搜索的问题: {total_with_search}")
        print(f"不需要搜索的问题: {total_without_search}")
        
        # 分类准确率
        if total_with_search > 0:
            search_accuracy = sum(1 for r in results if r["expected_search"] and r["correct"]) / total_with_search
            print(f"搜索问题准确率: {search_accuracy:.3f}")
        
        if total_without_search > 0:
            no_search_accuracy = sum(1 for r in results if not r["expected_search"] and r["correct"]) / total_without_search
            print(f"非搜索问题准确率: {no_search_accuracy:.3f}")
        
        return results

    def run_test(self, test_file, output_file):
        """运行测试"""
        # 加载测试数据
        test_data = self.load_test_data(test_file)
        print(f"加载了 {len(test_data)} 条测试数据")
        
        # 准备提示
        prompts = self.prepare_prompts(test_data)
        
        # 批量生成
        responses = self.batch_generate(prompts)
        
        # 评估结果
        results = self.evaluate_responses(test_data, responses)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"测试结果保存到 {output_file}")
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM批量测试")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./qwen_lora_output")
    parser.add_argument("--test_file", type=str, default="qwen_data_test.jsonl")
    parser.add_argument("--output_file", type=str, default="test_results.json")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.test_file):
        print(f"测试文件不存在: {args.test_file}")
        return
    
    if args.adapter_path and not os.path.exists(args.adapter_path):
        print(f"LoRA适配器路径不存在: {args.adapter_path}")
        print("将使用原始模型进行测试")
        args.adapter_path = None
    
    # 创建测试器
    tester = VLLMTester(args.model_path, args.adapter_path)
    
    # 运行测试
    results = tester.run_test(args.test_file, args.output_file)

if __name__ == "__main__":
    main()