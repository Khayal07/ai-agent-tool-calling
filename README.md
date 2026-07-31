# 🤖 AI Agent with Tool/Function Calling

İstifadəçi niyyətinə (**user intent**) əsasən xarici tool/funksiyaları dinamik şəkildə seçən, zəncirvari (**multi-step chaining**) icra edən və sonsuz dövrələrdən qorunan **LLM-based AI Agent**.

---

## 🚀 Features

* **🧩 Pydantic & Docstring ilə Tool Sxemləri**
  LLM-in yanlış tool seçməsinin və səhv parametr göndərməsinin qarşısını almaq üçün dəqiq müəyyən edilmiş tool sxemləri.

* **🔗 Multi-step Tool Calling**
  Bir neçə tool tələb edən mürəkkəb sorğuları məntiqi ardıcıllıqla analiz edir və tool-ları bir-birinin nəticəsindən istifadə edərək icra edir.

* **🛡️ Guardrail & Cost Protection**
  `.env` faylında idarə olunan `MAX_ITERATIONS` limiti sayəsində agentin nəzarətsiz şəkildə davam etməsinin və lazımsız API xərclərinin qarşısı alınır.

* **🧠 Reasoning & Execution Trace**
  Agentin icra prosesini izləmək və debug etmək üçün `Thought`, `Action` və `Observation` mərhələləri terminalda aydın şəkildə göstərilir.

* **⚙️ Dynamic Environment Configuration**
  Model adı, API Base URL, iteration limiti və digər parametrlər kodu dəyişmədən `.env` vasitəsilə idarə olunur.

---

## 🏗️ Agent Workflow

Agent istifadəçi sorğusunu qəbul etdikdən sonra aşağıdakı workflow üzrə işləyir:

```text
User Query
    │
    ▼
┌─────────────────┐
│  LLM / Reasoning │
└────────┬────────┘
         │
         ▼
   Select Tool
         │
         ▼
┌─────────────────┐
│   Tool Execution │
└────────┬────────┘
         │
         ▼
    Observation
         │
         ▼
   Need Another Tool?
      /        \
    YES         NO
     │           │
     ▼           ▼
  Next Tool   Final Response
```

Agent hər addımda əvvəlki tool-un nəticəsini nəzərə alaraq növbəti action-ı müəyyən edə bilir.

---

## 🛠️ Tech Stack

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| 🐍 Python           | Core programming language           |
| 🤖 OpenAI API       | LLM & tool/function calling         |
| 📦 Pydantic         | Tool schemas & parameter validation |
| 🔐 python-dotenv    | Environment configuration           |
| 🧰 Custom Tools     | External function execution         |
| 🔄 Function Calling | Dynamic tool selection & chaining   |

---

## 📂 Project Structure

```text
ai-agent-tool-calling/
│
├── agent.py
├── main.py
├── tools.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> Fayl strukturu layihənin implementasiyasına uyğun olaraq dəyişə bilər.

---

## 🛠️ Setup

### 1. Repository-ni klonlayın

```bash
git clone https://github.com/Khayal07/ai-agent-tool-calling.git
cd ai-agent-tool-calling
```

### 2. Virtual mühit yaradın

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Asılılıqları quraşdırın

```bash
pip install -r requirements.txt
```

### 4. Environment faylını konfiqurasiya edin

`.env.example` faylının adını `.env` olaraq dəyişin və API məlumatlarınızı daxil edin:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
MODEL_NAME=gpt-4o-mini
MAX_ITERATIONS=5
VERBOSE_LOGGING=True
```

> ⚠️ **Vacib:** `.env` faylınızı GitHub-a push etməyin. API açarınızı heç vaxt public repository-də paylaşmayın.

---

## 💻 Usage

Agent-i işə salmaq üçün:

```bash
python main.py
```

Daha sonra agentə müxtəlif səviyyəli sorğular verərək **single-step** və **multi-step tool calling** davranışını test edə bilərsiniz.

---

## 🧪 Example Query

Agent aşağıdakı kimi multi-step sorğunu emal edə bilər:

```text
Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir.
```

Bu sorğu üçün agent:

1. Cari location-u müəyyən edir.
2. Həmin şəhərin hava məlumatını əldə edir.
3. Temperaturu Celsius-dan Fahrenheit-ə çevirir.
4. Bütün nəticələri birləşdirərək final cavab yaradır.

---

## 📝 Example Execution Trace

```text
🚀 [SORĞU BAŞLADI]:
'Harada olduğuma görə hava necədir, sonra bu dərəcəni Fahrenheit-ə çevir.'

--- [ADDIM 1] ---
🛠️ [ACTION / TOOL CALL]: 'get_current_location'
📥 [PARAMETRLƏR]: {}

👁️ [OBSERVATION / RESULT]:
Baku


--- [ADDIM 2] ---
🛠️ [ACTION / TOOL CALL]: 'get_weather_by_city'
📥 [PARAMETRLƏR]: {'city': 'Baku'}

👁️ [OBSERVATION / RESULT]:
22°C, Günəşli


--- [ADDIM 3] ---
🛠️ [ACTION / TOOL CALL]: 'convert_celsius_to_fahrenheit'
📥 [PARAMETRLƏR]: {'celsius': 22.0}

👁️ [OBSERVATION / RESULT]:
22.0°C dərəcə 71.6°F-ə bərabərdir.


--- [FINAL RESPONSE] ---

Hazırda olduğunuz Bakı şəhərində hava 22°C və günəşlidir.
Bu temperatur Fahrenheit vahidi ilə 71.6°F-ə bərabərdir.
```

---

## 🛡️ Guardrail & Iteration Protection

Agentin sonsuz tool-calling loop-a düşməsinin qarşısını almaq üçün `MAX_ITERATIONS` limiti tətbiq olunur.

Məsələn:

```env
MAX_ITERATIONS=5
```

Bu o deməkdir ki, agent maksimum **5 iteration** ərzində tool çağırışlarını davam etdirə bilər.

Limitə çatdıqda agent icranı dayandırır və nəzarətsiz API istifadəsinin qarşısını alır.

---

## 🔍 Key Concepts Demonstrated

Bu layihə aşağıdakı əsas AI Engineering konseptlərini nümayiş etdirir:

* LLM-based Agents
* OpenAI Function / Tool Calling
* Dynamic Tool Selection
* Multi-step Agentic Workflows
* Tool Chaining
* Pydantic Schema Validation
* Environment-based Configuration
* Execution Tracing
* Guardrails
* Iteration & Cost Protection
* Agent Debugging

---

## 🎯 Project Goal

Layihənin əsas məqsədi sadə bir LLM chatbot-dan fərqli olaraq, **real tool-lardan istifadə edə bilən, nəticələri analiz edib növbəti action-ı müəyyənləşdirən və mürəkkəb istifadəçi sorğularını bir neçə addımda həll edə bilən AI Agent** arxitekturasını nümayiş etdirməkdir.

---

## 👨‍💻 Author

**AI Engineering Project**

Built with Python, LLMs, Tool Calling and Agentic AI concepts.
