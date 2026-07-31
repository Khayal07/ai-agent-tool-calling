import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

# .env faylından mühit dəyişənlərini yükləyirik
load_dotenv()

class ChainedToolAgent:
    """
    Zəncirvari (Multi-step) Tool çağırışlarını dəstəkləyən və 
    bütün konfiqurasiyaları .env faylından oxuyan Agent sinfi.
    """
    def __init__(self, model_name: str = None, base_url: str = None, max_iterations: int = None):
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Bütün konfiqurasiyaları .env-dən oxuyuruq (Fallback mexanizmi ilə)
        selected_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        selected_base_url = base_url or os.getenv("MODEL_BASE_URL", None)
        
        env_max_iter = os.getenv("MAX_ITERATIONS", "5")
        self.max_iterations = max_iterations or int(env_max_iter)
        
        llm_kwargs = {
            "model": selected_model,
            "temperature": 0
        }
        if selected_base_url:
            llm_kwargs["base_url"] = selected_base_url
            
        self.llm = ChatOpenAI(**llm_kwargs)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, user_query: str) -> str:
        """
        Ardıcıl tool çağırışlarını zəncirvari rejimdə idarə edən dövr.
        """
        messages = [HumanMessage(content=user_query)]
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # Əgər LLM növbəti adımda tool çağırmaq istəyirsə:
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    # Tool-u tapıb icra edirik
                    selected_tool = self.tools_by_name[tool_name]
                    tool_output = selected_tool.invoke(tool_args)

                    # Nəticəni konversasiya tarixçəsinə əlavə edirik
                    messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call_id
                        )
                    )
            else:
                # LLM daha tool tələb etmirsə, yekun cavabı qaytarırıq
                return ai_msg.content

        return "Maksimum iterasiya limitinə çatıldı."


if __name__ == "__main__":
    agent = ChainedToolAgent()
    
    # Checkpoint 4 üçün zəncirvari test sorğusu
    query = "Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir."
    print(f"İstifadəçi sorğusu: {query}\n" + "=" * 60)
    
    final_answer = agent.run(query)
    print(f"\nAgent Yekun Cavabı:\n{final_answer}")