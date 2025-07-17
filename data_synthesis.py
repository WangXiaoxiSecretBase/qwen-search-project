#!/usr/bin/env python3
"""
使用DeepSeek-R1-Distill-Qwen-7B合成训练数据
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import os

class DataSynthesizer:
    def __init__(self, model_path="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 搜索工具定义
        self.search_tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "搜索引擎，在需要获取实时信息、最新数据、具体事实查询时需要调用此工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"},
                        "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
                    },
                    "required": ["keyword"]
                }
            }
        }
        
        # 初始化FakeSearch
        from fake_search import FakeSearch
        self.fake_search = FakeSearch()

    def generate_response(self, question):
        """始终使用带工具描述的system_prompt"""
        system_prompt = f"""你是一个智能助手，可以使用搜索工具获取信息。当需要实时信息、最新数据或具体事实时，请使用搜索工具。\n可用工具：\n{json.dumps(self.search_tool, ensure_ascii=False, indent=2)}\n请仔细思考用户的问题，判断是否需要使用搜索工具。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response

    def extract_tool_calls(self, response):
        """提取工具调用"""
        # 简单的工具调用提取逻辑
        if "search" in response and "{" in response:
            # 这里可以实现更复杂的工具调用解析
            return True
        return False

    def simulate_tool_response(self, response, question):
        """模拟工具调用并生成完整回复"""
        if not self.extract_tool_calls(response):
            return response
        
        # 提取搜索关键词（简化版）
        import re
        keyword_match = re.search(r'"keyword":\s*"([^"]+)"', response)
        if keyword_match:
            keyword = keyword_match.group(1)
            search_results = self.fake_search.search(keyword, 3)
            
            # 构造包含搜索结果的新对话
            search_result_text = "\n".join([f"搜索结果{i+1}: {result}" for i, result in enumerate(search_results)])
            
            follow_up_messages = [
                {"role": "system", "content": "你是一个智能助手，请基于搜索结果回答用户问题。"},
                {"role": "user", "content": f"问题: {question}\n\n搜索结果:\n{search_result_text}\n\n请基于这些搜索结果回答问题。"}
            ]
            
            text = self.tokenizer.apply_chat_template(
                follow_up_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            final_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return final_response
        
        return response

    def load_questions(self, *question_files):
        """加载所有问题文件，合并为一个列表"""
        all_questions = []
        for file in question_files:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    all_questions.extend([line.strip() for line in f if line.strip()])
        return all_questions

    def synthesize_data(self, questions, output_file):
        """合成训练数据，模型自行判断是否调用search"""
        synthetic_data = []
        print("处理所有问题...")
        for question in tqdm(questions):
            try:
                response = self.generate_response(question)
                if self.extract_tool_calls(response):
                    final_response = self.simulate_tool_response(response, question)
                    synthetic_data.append({
                        "question": question,
                        "response": final_response,
                        "use_search": True
                    })
                else:
                    synthetic_data.append({
                        "question": question,
                        "response": response,
                        "use_search": False
                    })
            except Exception as e:
                print(f"处理问题时出错: {question[:50]}... 错误: {e}")
                continue
        random.shuffle(synthetic_data)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
        print(f"合成完成！共生成 {len(synthetic_data)} 条数据，保存到 {output_file}")
        return synthetic_data

if __name__ == "__main__":
    synthesizer = DataSynthesizer()
    # 合并加载所有问题
    all_questions = synthesizer.load_questions(
        "question_with_search.txt",
        "question_without_search.txt"
    )
    # 合成数据
    synthetic_data = synthesizer.synthesize_data(
        all_questions[:200],  # 可调整数量
        "synthetic_data.json"
    )