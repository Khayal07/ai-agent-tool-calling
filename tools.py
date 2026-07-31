from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ==========================================
# 1. TOOL: Məkan Məlumatının Alınması
# ==========================================
@tool
def get_current_location() -> str:
    """
    İstifadəçinin hazırda olduğu coğrafi məkanı (şəhər adını) qaytarır.
    
    QAYDA: İstifadəçi sorğusunda konkrеt şəhər adı çəkmədikdə (məsələn: "burada hava necədir?", 
    "haradayam?") İLK ÖNCƏ bu tool çağırılmalıdır.
    
    Returns:
        str: Cari şəhərin adı.
    """
    # Real ssenaride IP/GPS API istifade oluna biler. Test ucun sabit deyer:
    return "Baku"
    

# ==========================================
# 2. TOOL: Hava Məlumatının Alınması
# ==========================================
class WeatherInput(BaseModel):
    city: str = Field(
        description="Hava məlumatı alınacaq dəqiq şəhər adı (məs: 'Baku', 'London', 'Istanbul')."
    )
    
@tool(args_schema=WeatherInput)
def get_weather_by_city(city: str) -> str:
    """
    Qeyd olunan konkrеt şəhər üçün cari hava temperaturunu Selsi (°C) ilə qaytarır.
    
    QAYDA: 
    1. Əgər şəhər adı bəlli deyilsə, Bu tool-u çağırmadan öncə `get_current_location` istifadə edilməlidir.
    2. Bu tool yalnız temperatur almaq üçündür, dərəcə çevrilməsi (Fahrenheit-ə) ETMİR.
    
    Args:
        city (str): Şəhər adı.
        
    Returns:
        str: Selsi ilə temperatur və hava şəraiti.
    """
    # Model/Mock verilənlər bazası:
    weather_data = {
        "baku": "22°C, Günəşli",
        "london": "15°C, Yağışlı",
        "istanbul": "20°C, Buludlu"
    }
    city_clean = city.lower().strip()
    return weather_data.get(city_clean, f"21°C, Aydın hava ({city})")


# ==========================================
# 3. TOOL: Temperatur Çevrilməsi
# ==========================================
class TemperatureConversionInput(BaseModel):
    celsius: float = Field(
        description="Fahrenheit-ə çevriləcək dərəcə dəyəri (Selsi ilə, məsələn: 22.0)."
    )

@tool(args_schema=TemperatureConversionInput)
def convert_celsius_to_fahrenheit(celsius: float) -> str:
    """
    Selsi (°C) dərəcəsini Fahrenheit (°F) vahidinə çevirir.
    
    QAYDA: Yalnız və yalnız istifadəçi xüsusi olaraq dərəcənin Fahrenheit-ə çevrilməsini 
    tələb etdikdə çağırılmalıdır. Sadə hava sorğularında bu tool-u çağırmayın.
    
    Args:
        celsius (float): Selsi ilə dərəcə sayı.
        
    Returns:
        str: Fahrenheit ilə hesablanmış nəticə.
    """
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C dərəcə {fahrenheit:.1f}°F dərəcəyə bərabərdir."