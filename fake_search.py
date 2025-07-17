#!/usr/bin/env python3
"""
网络搜索工具模拟
"""

from openai import OpenAI
import requests

class FakeSearch:
    def __init__(self, base_url="http://127.0.0.1:8000/v1"):
        # 尽量自己部署模型用于实验，可以是任何模型
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        try:
            self.model = self.client.models.list().data[0].id
        except:
            self.model = "default"

    def chat(self, messages: list):
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=12000,
                temperature=0.5,
                n=1
            )
            return result
        except Exception as e:
            print(f"API调用失败: {e}")
            # 返回模拟结果
            class MockResult:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockResult(f"关于'{messages[0]['content']}'的搜索结果模拟")

    def search(self, keyword, top_k=3):
        res = self.chat([{
            "role": "user",
            "content": f"请你扮演一个搜索引擎，对于任何的输入信息，给出 {min(top_k, 10)} 个合理的搜索结果，以列表的方式呈现。列表由空行分割，每行的内容是不超过500字的搜索结果。\n\n输入: {keyword}"
        }])
        
        try:
            content = res.choices[0].message.content.split("</think>")[-1].strip()
            res_list = content.split("\n")
            return [res.strip() for res in res_list if len(res.strip()) > 0][:top_k]
        except:
            # 备用搜索结果
            return [
                f"关于'{keyword}'的搜索结果1：这是一个模拟的搜索结果，包含了相关的信息。",
                f"关于'{keyword}'的搜索结果2：这是另一个模拟的搜索结果，提供了不同的视角。",
                f"关于'{keyword}'的搜索结果3：这是第三个模拟的搜索结果，补充了更多细节。"
            ][:top_k]

if __name__ == "__main__":
    import sys
    search = FakeSearch()
    if len(sys.argv) > 1:
        results = search.search(sys.argv[1], 5)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
    else:
        print("用法: python fake_search.py <关键词>")