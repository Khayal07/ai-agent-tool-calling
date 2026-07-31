from agent import TraceableAgent

def main():
    # .env faylından konfiqurasiyaları oxuyan Agent-i başlatırıq
    agent = TraceableAgent()

    print("==================================================")
    print("🤖 AI AGENT TOOL CALLING - TEST RUNNER")
    print("==================================================")

    # TEST 1: Zəncirvari (Multi-step) Sorğu
    print("\n>>> TEST 1: Zəncirvari Sorğu (Məkan -> Hava -> Dərəcə çevrilməsi)")
    agent.run("Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir.")

    # TEST 2: Birbaşa Tool Seçimi
    print("\n>>> TEST 2: Yalnız Dərəcə Çevrilməsi")
    agent.run("30 dərəcə Selsi neçə Fahrenheit edir?")

    # TEST 3: Tool Tələb ETMƏYƏN Sorğu (Trick Check)
    print("\n>>> TEST 3: Ümumi Bilik Sorğusu (Tool çağırılmamalıdır)")
    agent.run("Süni intellekt nədir?")

if __name__ == "__main__":
    main()