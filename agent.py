import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit

# .env faylından API key-i yükləyirik
load_dotenv()

class ToolSelectorAgent:
    """
    İstifadəçinin müxtəlif ifadə formalarındakı sorğularına əsasən
    düzgün tool seçimi edən LLM Agent sinfi.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Mövcud tool-ların siyahısı
        self.tools = [get_current_location, get_weather_by_city, convert_celsius_to_fahrenheit]
        
        # LLM-in inisializasiyası (nəticələrin determinik olması üçün temperature=0)
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Tool-ları LLM-ə bağlayırıq (Function Calling)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def analyze_query_tool_selection(self, query: str) -> Dict[str, Any]:
        """
        Gələn sorğunu LLM-ə göndərir və LLM-in tool seçib-seçmədiyini müəyyənləşdirir.
        """
        response = self.llm_with_tools.invoke(query)
        
        # LLM tool çağırmaq qərarına gəlibmi?
        if response.tool_calls:
            selected_tools = [
                {"tool_name": call["name"], "arguments": call["args"]}
                for call in response.tool_calls
            ]
            return {
                "query": query,
                "requires_tool": True,
                "selected_tools": selected_tools
            }
        else:
            return {
                "query": query,
                "requires_tool": False,
                "direct_response": response.content
            }


# Müxtəlif ifadə formalarında test ssenarisi
if __name__ == "__main__":
    agent = ToolSelectorAgent()

    test_queries = [
        # 1. Müəyyən olunmuş şəhər ilə doğrudan hava sorğusu -> (get_weather_by_city)
        "London şəhərində hava temperaturu neçə dərəcədir?",
        
        # 2. Qeyri-müəyyən/Məkan bilinməyən sorğu -> (get_current_location)
        "Hazırda olduğum yerdə hava necədir?",
        
        # 3. Yalnız çevrilmə tələb edən sorğu -> (convert_celsius_to_fahrenheit)
        "25 dərəcə Selsi neçə Fahrenheit edir?",
        
        # 4. Trick Check: Tool tələb ETMƏYƏN sadə sorğu -> (Birbaşa cavab, heç bir tool seçilməməlidir)
        "Fransanın paytaxtı haradır və AI nədir?"
    ]

    print("=== TOOL SEÇİMİ TEST NƏTİCƏLƏRİ ===\n")
    for q in test_queries:
        res = agent.analyze_query_tool_selection(q)
        print(f"Sorğu: '{res['query']}'")
        if res["requires_tool"]:
            print(f" Seçilən Tool(lar): {res['selected_tools']}\n")
        else:
            print(f" Tool İstifadə Olunmadı (Birbaşa Cavab): {res['direct_response'][:60]}...\n")