#!/usr/bin/env python3
"""
交互式演示程序，支持多轮对话和工具调用
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from fake_search import FakeSearch

class QwenDemo:
    def __init__(self, model_path="/remote-home1/share/models/Qwen2.5-0.5B-Instruct", 
                 adapter_path="./qwen_lora_output"):
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        try:
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            print("成功加载LoRA适配器")
        except Exception as e:
            print(f"无法加载LoRA适配器: {e}")
            print("使用原始模型")
            self.model = self.base_model
        
        # 初始化搜索工具
        self.search_tool = FakeSearch()
        
        # 搜索工具定义
        self.tool_definition = {
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
        
        # 对话历史
        self.conversation_history = []

    def extract_tool_call(self, response):
        """提取工具调用"""
        # 查找工具调用模式
        tool_pattern = r'{"name":\s*"search",\s*"parameters":\s*{[^}]*}}'
        matches = re.findall(tool_pattern, response)
        
        if matches:
            try:
                tool_call = json.loads(matches[0])
                return tool_call
            except json.JSONDecodeError:
                pass
        
        # 尝试更灵活的匹配
        if "search" in response and "keyword" in response:
            keyword_match = re.search(r'"keyword":\s*"([^"]+)"', response)
            if keyword_match:
                keyword = keyword_match.group(1)
                top_k_match = re.search(r'"top_k":\s*(\d+)', response)
                top_k = int(top_k_match.group(1)) if top_k_match else 3
                
                return {
                    "name": "search",
                    "parameters": {
                        "keyword": keyword,
                        "top_k": top_k
                    }
                }
        
        return None

    def call_tool(self, tool_call):
        """调用工具"""
        if tool_call["name"] == "search":
            params = tool_call["parameters"]
            keyword = params.get("keyword", "")
            top_k = params.get("top_k", 3)
            
            print(f"🔍 搜索关键词: {keyword}")
            
            try:
                search_results = self.search_tool.search(keyword, top_k)
                return {
                    "name": "search",
                    "content": "\n".join([f"结果{i+1}: {result}" for i, result in enumerate(search_results)])
                }
            except Exception as e:
                return {
                    "name": "search",
                    "content": f"搜索失败: {str(e)}"
                }
        
        return None

    def generate_response(self, user_input, max_turns=3):
        """生成回复，支持多轮工具调用"""
        # 构建系统提示
        system_prompt = f"""你是一个智能助手，可以使用搜索工具获取信息。当需要实时信息、最新数据或具体事实时，请使用搜索工具。

可用工具：
{json.dumps(self.tool_definition, ensure_ascii=False, indent=2)}

请仔细思考用户的问题，判断是否需要使用搜索工具。如果需要使用工具，请按照以下格式调用：
{{"name": "search", "parameters": {{"keyword": "搜索关键词", "top_k": 3}}}}"""
        
        # 构建对话历史
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        for turn in range(max_turns):
            # 生成回复
            text = self.tokenizer.apply_chat_template(
                messages,
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 检查是否有工具调用
            tool_call = self.extract_tool_call(response)
            
            if tool_call:
                print(f"🤖 助手想要调用工具: {tool_call['name']}")
                
                # 调用工具
                tool_result = self.call_tool(tool_call)
                
                if tool_result:
                    # 将工具调用和结果添加到对话历史
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["name"]
                    })
                    
                    # 继续生成基于工具结果的回复
                    continue
            else:
                # 没有工具调用，返回最终回复
                return response.strip()
        
        # 如果达到最大轮数，返回最后的回复
        return response.strip()

    def chat(self):
        """开始聊天"""
        print("=== Qwen助手 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清除对话历史")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("对话历史已清除")
                    continue
                
                if not user_input:
                    continue
                
                # 生成回复
                print("\n🤖 助手: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # 更新对话历史
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # 限制对话历史长度
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n❌ 出现错误: {e}")
                continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen交互式演示")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./qwen_lora_output")
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = QwenDemo(args.model_path, args.adapter_path)
    
    # 开始聊天
    demo.chat()

if __name__ == "__main__":
    main()