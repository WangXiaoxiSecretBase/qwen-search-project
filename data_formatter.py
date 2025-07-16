#!/usr/bin/env python3
"""
将合成的数据转换为Qwen格式的训练数据
"""

import json
import random
from typing import List, Dict

class DataFormatter:
    def __init__(self):
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

    def format_for_qwen(self, data: List[Dict]) -> List[Dict]:
        """将数据格式化为Qwen训练格式"""
        formatted_data = []
        
        for item in data:
            question = item["question"]
            response = item["response"]
            use_search = item.get("use_search", False)
            
            # 构造系统提示
            if use_search:
                system_content = f"""你是一个智能助手，可以使用搜索工具获取信息。当需要实时信息、最新数据或具体事实时，请使用搜索工具。

可用工具：
{json.dumps(self.search_tool, ensure_ascii=False, indent=2)}

请仔细思考用户的问题，判断是否需要使用搜索工具。"""
            else:
                system_content = "你是一个智能助手，请仔细思考用户的问题并给出详细回答。"
            
            # 构造对话格式
            conversation = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            
            formatted_data.append({
                "messages": conversation,
                "use_search": use_search
            })
        
        return formatted_data

    def add_thinking_wrapper(self, response: str) -> str:
        """为回复添加思考标签"""
        if "<think>" in response or "</think>" in response:
            return response
        
        # 简单的思考内容生成
        thinking_content = "让我思考一下这个问题..."
        return f"<think>\n{thinking_content}\n</think>\n\n{response}"

    def split_train_test(self, data: List[Dict], train_ratio: float = 0.8) -> tuple:
        """划分训练集和测试集"""
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]

    def save_jsonl(self, data: List[Dict], filename: str):
        """保存为JSONL格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_data(self, input_file: str, output_prefix: str = "qwen_data"):
        """处理数据的主函数"""
        # 加载原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"加载了 {len(raw_data)} 条原始数据")
        
        # 格式化为Qwen格式
        formatted_data = self.format_for_qwen(raw_data)
        
        # 限制数据量并划分训练测试集
        if len(formatted_data) > 385:
            formatted_data = formatted_data[:385]
        
        train_data, test_data = self.split_train_test(formatted_data)
        
        # 进一步限制训练数据
        if len(train_data) > 300:
            train_data = train_data[:300]
        if len(test_data) > 85:
            test_data = test_data[:85]
        
        # 保存数据
        self.save_jsonl(train_data, f"{output_prefix}_train.jsonl")
        self.save_jsonl(test_data, f"{output_prefix}_test.jsonl")
        
        print(f"训练集: {len(train_data)} 条数据 -> {output_prefix}_train.jsonl")
        print(f"测试集: {len(test_data)} 条数据 -> {output_prefix}_test.jsonl")
        
        # 统计信息
        train_with_search = sum(1 for item in train_data if item.get("use_search", False))
        test_with_search = sum(1 for item in test_data if item.get("use_search", False))
        
        print(f"训练集中需要搜索的数据: {train_with_search}")
        print(f"测试集中需要搜索的数据: {test_with_search}")
        
        return train_data, test_data

if __name__ == "__main__":
    formatter = DataFormatter()
    train_data, test_data = formatter.process_data("synthetic_data.json")