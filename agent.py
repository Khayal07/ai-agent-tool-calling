import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

load_dotenv()

class ExecutableToolAgent:
    """
    Tool-ları dinamik icra edən və nəticəni LLM-ə qaytararaq 
    təbii dildə yekun cavab generasiya edən Agent.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        # Tool adlarını onların obyekti ilə uyğunlaşdıran map
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, user_query: str) -> str:
        """
        İstifadəçi sorğusunu qəbul edir, lazım gəldikdə tool-u icra edir,
        nəticəni LLM-ə geri verir və təbii dildə yekun cavab qaytarır.
        """
        messages = [HumanMessage(content=user_query)]
        
        # 1. LLM-ə ilkin sorğu
        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # 2. Əgər LLM tool çağırılması tələb edirsə:
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                # İcra olunacaq tool-u tapırıq və çağırırıq
                selected_tool = self.tools_by_name[tool_name]
                tool_output = selected_tool.invoke(tool_args)
                
                # 3. Tool nəticəsini ToolMessage kimi tarixçəyə əlavə edirik
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call_id
                    )
                )
            
            # 4. Yekun təbii cavab almaq üçün LLM-i yenidən çağırırıq
            final_response = self.llm_with_tools.invoke(messages)
            return final_response.content
        else:
            # Tool lazım olmadıqda birbaşa cavab
            return ai_msg.content


if __name__ == "__main__":
    agent = ExecutableToolAgent()
    
    # Test sorğusu
    query = "London şəhərində hava necədir?"
    print(f"İstifadəçi sorğusu: {query}")
    print("-" * 50)
    final_answer = agent.run(query)
    print(f"Agent Cavabı: {final_answer}")