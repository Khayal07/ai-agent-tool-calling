import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

# .env faylından dəyişənləri yükləyirik
load_dotenv()

class ExecutableToolAgent:
    """
    Tool-ları dinamik icra edən, model parametrlərini .env faylından oxuyan 
    və nəticəni LLM-ə qaytararaq təbii dildə yekun cavab generasiya edən Agent.
    """
    def __init__(self, model_name: str = None, base_url: str = None):
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Prioritet: Kodda verilən parametr > .env faylındakı dəyər > Susmaya görə (default) dəyər
        selected_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        selected_base_url = base_url or os.getenv("MODEL_BASE_URL", None)
        
        # LLM parametrlərinin dinamik konfiqurasiyası
        llm_kwargs = {
            "model": selected_model,
            "temperature": 0
        }
        
        # Əgər .env-də custom base_url varsa, onu əlavə edirik
        if selected_base_url:
            llm_kwargs["base_url"] = selected_base_url
            
        self.llm = ChatOpenAI(**llm_kwargs)
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
    
    query = "London şəhərində hava necədir?"
    print(f"İstifadəçi sorğusu: {query}")
    print("-" * 50)
    final_answer = agent.run(query)
    print(f"Agent Cavabı: {final_answer}")