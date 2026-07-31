import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

# .env faylından mühit dəyişənlərini yükləyirik
load_dotenv()

class AgentLogger:
    """
    Agent-in reasoning (düşüncə), tool çağırışı və observation (nəticə) 
    addımlarını terminalda aydın loglayan köməkçi sinif.
    """
    @staticmethod
    def log_step(step: int, reasoning: str):
        print(f"\n--- [ADDIM {step}] ---")
        if reasoning:
            print(f"[THOUGHT / REASONING]: {reasoning}")

    @staticmethod
    def log_tool_call(tool_name: str, args: dict):
        print(f"[ACTION / TOOL CALL]: '{tool_name}' | Parametrlər: {args}")

    @staticmethod
    def log_observation(output: str):
        print(f"[OBSERVATION / RESULT]: {output}")

    @staticmethod
    def log_final_response(response: str):
        print(f"\n[FINAL RESPONSE]:\n{response}\n" + "=" * 60)


class TraceableAgent:
    """
    Bütün reasoning və tool-calling izlərini aydın loglayan 
    və parametrləri .env faylından oxuyan Agent sinfi.
    """
    def __init__(
        self, 
        model_name: str = None, 
        base_url: str = None, 
        max_iterations: int = None,
        verbose: bool = None
    ):
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Bütün konfiqurasiyalar .env faylından oxunur
        selected_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        selected_base_url = base_url or os.getenv("MODEL_BASE_URL", None)
        
        env_max_iter = os.getenv("MAX_ITERATIONS", "5")
        self.max_iterations = max_iterations or int(env_max_iter)
        
        env_verbose = os.getenv("VERBOSE_LOGGING", "True").lower() in ("true", "1", "yes")
        self.verbose = verbose if verbose is not None else env_verbose

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
        Sorğunu icra edir və debug üçün hər bir addımın izini (trace) loglayır.
        """
        messages = [HumanMessage(content=user_query)]
        iterations = 0

        if self.verbose:
            print(f"\n[SORĞU BAŞLADI]: '{user_query}'")

        while iterations < self.max_iterations:
            iterations += 1
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if self.verbose:
                AgentLogger.log_step(iterations, ai_msg.content)

            # Əgər tool çağırışı varsa:
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    if self.verbose:
                        AgentLogger.log_tool_call(tool_name, tool_args)

                    selected_tool = self.tools_by_name[tool_name]
                    tool_output = selected_tool.invoke(tool_args)

                    if self.verbose:
                        AgentLogger.log_observation(str(tool_output))

                    messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call_id
                        )
                    )
            else:
                # Yekun cavab alındıqda
                if self.verbose:
                    AgentLogger.log_final_response(ai_msg.content)
                return ai_msg.content

        warning_msg = (
            f"[XƏBƏRDARLIQ] Təhlükəsizlik limiti: Maksimum təkrarlanma limitinə ({self.max_iterations}) çatıldı! "
            "Agent sonsuz dövrün qarşısını almaq üçün nəzarətli şəkildə dayandırıldı."
        )
        if self.verbose:
            print(f"\n⚠️ {warning_msg}")
        return warning_msg


if __name__ == "__main__":
    agent = TraceableAgent()
    
    # Reasoning və Execution trace-i görmək üçün zəncirvari test sorğusu
    query = "Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir."
    agent.run(query)