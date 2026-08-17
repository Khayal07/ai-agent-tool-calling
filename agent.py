import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import ValidationError
from tools import (
    get_current_location,
    get_weather_by_coordinates,
    convert_celsius_to_fahrenheit,
)

# ==========================================
# SİSTEM PROMPTU: Tool seçimi qaydaları
# ==========================================
SYSTEM_PROMPT = """Sən bir AI Agent'sən. İstifadəçinin sorğusunu analiz edib ən uyğun tool-u seçirsən və ya birbaşa cavab verirsən.

MÖVCUD TOOL-LAR:
1. get_current_location(city=None) — Məkan koordinatlarını alır.
   - İstifadəçi 'burada', 'haradayam', 'cari yerimdə' kimi ifadə işlədibsə: HEÇ BİR parametr olmadan çağır (IP ünvanına görə işləyir).
   - İstifadəçi konkret şəhər adı çəkibsə (məsələn 'Parijdə hava necədir?'): həmin şəhəri 'city' parametri ilə ötür (geocoding).
2. get_weather_by_coordinates(latitude, longitude) — Verilmiş koordinatlardakı havanı °C ilə qaytarır.
   - YALNIZ `get_current_location` nəticəsindən koordinatlar məlum olduqda çağır. Koordinatları heç vaxt özün təxmin etmə!
3. convert_celsius_to_fahrenheit(celsius) — °C-ni °F-ə çevirir.
   - YALNIZ istifadəçi açıq şəkildə Fahrenheit-ə çevrilmə istədikdə çağır.

SEÇİM QAYDALARI (Trick Check):
- Ümumi bilik/mühakimə sorğuları ('Süni intellekt nədir?', '2+2 neçədir?', tarix sualları) tool TƏLƏB ETMİR → birbaşa qısa, aydın cavab ver.
- Yalnız çevrilmə sorğusu ('30 dərəcə Selsi neçə Fahrenheit edir?') → YALNIZ convert tool-u; hava tool-u çağırma.
- Şəhər + hava sorğusu ('Bakıda hava necədir?') → get_current_location(city='Baku') → get_weather_by_coordinates.
- Cari yer + hava sorğusu ('burada hava necədir?') → get_current_location() → get_weather_by_coordinates.
- Hava + Fahrenheit sorğusu → üç addım: location → weather → convert.
- Məlumat əldə etməzdən əvvəl tool çağırmağa ehtiyac yoxdursa, tool çağırma; birbaşa cavab ver."""

# ==========================================
# GUARDRAIL: Sonsuz dövr qoruması
# ==========================================
MAX_ITERATIONS = 5

# .env faylından mühit dəyişənlərini yükləyirik
load_dotenv()

class AgentLogger:
    """
    Agent-in reasoning (düşüncə), tool çağırışı və observation (nəticə)
    addımlarını strukturlaşdırılmış formada terminala loglayan köməkçi sinif.

    Format: [Iteration X] Model Thought -> Tool Selected -> Arguments -> Tool Output
    """
    @staticmethod
    def _format_args(args: dict) -> str:
        if not args:
            return "{}"
        return ", ".join(f"{key}={value}" for key, value in args.items())

    @staticmethod
    def log_iteration(
        step: int,
        reasoning: str,
        tool_name: str = None,
        args: dict = None,
        output: str = None,
    ):
        line = f"[Iteration {step}] Model Thought: {reasoning if reasoning else '(yoxdur)'}"
        if tool_name:
            line += f" -> Tool Selected: {tool_name} -> Arguments: {AgentLogger._format_args(args)}"
        if output is not None:
            line += f" -> Tool Output: {output}"
        print(line)

    @staticmethod
    def log_final_response(step: int, response: str):
        print(f"[Iteration {step}] Final Answer: {response}")
        print("=" * 60)


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
        self.tools = [get_current_location, get_weather_by_coordinates, convert_celsius_to_fahrenheit]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        
        # Bütün konfiqurasiyalar .env faylından oxunur
        selected_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        selected_base_url = base_url or os.getenv("MODEL_BASE_URL", None)
        
        env_max_iter = os.getenv("MAX_ITERATIONS", str(MAX_ITERATIONS))
        try:
            env_max_iter_value = int(env_max_iter)
        except ValueError:
            env_max_iter_value = MAX_ITERATIONS
        self.max_iterations = max_iterations or env_max_iter_value
        if self.max_iterations < 1:
            self.max_iterations = MAX_ITERATIONS
        
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
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]
        iterations = 0

        if self.verbose:
            print(f"\n[SORĞU BAŞLADI]: '{user_query}'")

        while iterations < self.max_iterations:
            iterations += 1
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # Əgər tool çağırışı varsa:
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    try:
                        selected_tool = self.tools_by_name[tool_name]
                        tool_output = selected_tool.invoke(tool_args)
                        result_payload = (
                            f"[TOOL OUTPUT] Tool: '{tool_name}' | Nəticə: {tool_output}"
                        )
                    except KeyError:
                        result_payload = (
                            f"[TOOL ERROR] '{tool_name}' tool-u mövcud deyil. "
                            f"Mövcud tool-lar: {', '.join(self.tools_by_name.keys())}."
                        )
                    except ValidationError as exc:
                        result_payload = (
                            f"[TOOL ERROR] '{tool_name}' tool-u üçün parametrlər yanlışdır: {exc}."
                        )
                    except (ConnectionError, RuntimeError, ValueError, TimeoutError) as exc:
                        result_payload = (
                            f"[TOOL ERROR] '{tool_name}' tool-u icra edilərkən xəta baş verdi: {exc}."
                        )
                    except Exception as exc:
                        result_payload = (
                            f"[TOOL ERROR] '{tool_name}' tool-u gözlənilməz xəta ilə qarşılaşdı: {exc}."
                        )

                    if self.verbose:
                        AgentLogger.log_iteration(
                            iterations, ai_msg.content, tool_name, tool_args, result_payload
                        )

                    messages.append(
                        ToolMessage(
                            content=result_payload,
                            tool_call_id=tool_call_id
                        )
                    )
            else:
                # Yekun cavab alındıqda
                if self.verbose:
                    AgentLogger.log_final_response(iterations, ai_msg.content)
                return ai_msg.content

        warning_msg = (
            f"Üzr istəyirik! Sorğunuzu həll etmək üçün lazım olan addımların sayı "
            f"təhlükəsizlik limitini ({self.max_iterations} iterasiya) keçdi. "
            "Agent sonsuz dövrə düşməmək və nəzarətsiz API xərclərinin qarşısını "
            "almaq üçün nəzarətli şəkildə dayandırıldı."
        )
        if self.verbose:
            print(f"\n⚠️ {warning_msg}")
        return warning_msg


if __name__ == "__main__":
    agent = TraceableAgent()
    
    # Reasoning və Execution trace-i görmək üçün zəncirvari test sorğusu
    query = "Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir."
    agent.run(query)