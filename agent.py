import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

# .env faylından mühit dəyişənlərini yükləyirik
load_dotenv()

class SafeLoopAgent:
    """
    Sonsuz dövrə və nəzarətsiz API xərclərinə qarşı maksimum təkrarlanma 
    limiti (Max Iteration Guardrail) ilə qorunan Agent sinfi.
    """
    def __init__(self, model_name: str = None, base_url: str = None, max_iterations: int = None):
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Bütün parametr konfiqurasiyaları .env faylından oxunur
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
        Maksimum dövr nəzarəti (Infinite Loop Guardrail) ilə sorğunu icra edir.
        """
        messages = [HumanMessage(content=user_query)]
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # Əgər LLM tool çağırmaq istəyirsə:
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    selected_tool = self.tools_by_name[tool_name]
                    tool_output = selected_tool.invoke(tool_args)

                    messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call_id
                        )
                    )
            else:
                # Zəncir uğurla bitdikdə və LLM yekun cavabı verdikdə
                return ai_msg.content

        # Əgər MAX_ITERATIONS limitinə çatılarsa (Sonsuz dövrün qarşısını almaq üçün məcburi dayandırma)
        return (
            f"[XƏBƏRDARLIQ] Təhlükəsizlik limiti: Maksimum təkrarlanma limitinə ({self.max_iterations}) çatıldı! "
            "Sonsuz dövrün və nəzarətsiz API xərclərinin qarşısını almaq üçün agent prosesi nəzarətli şəkildə saxladı."
        )


if __name__ == "__main__":
    # Test: Qəsdən aşağı limit (max_iterations=1) qoyaraq qoruma sistemini test edirik
    test_agent = SafeLoopAgent(max_iterations=1)
    
    # 2 addım tələb edən sorğu göndəririk ki, 1-ci adımda limitə çatıb qorumanı işə salsın
    query = "London üçün hava necədir və bunu Fahrenheit-ə çevir?"
    print(f"Test Sorğusu: {query}")
    print(f"Təyin olunmuş Max Iteration: 1")
    print("-" * 50)
    
    response = test_agent.run(query)
    print(f"Agent Cavabı:\n{response}")