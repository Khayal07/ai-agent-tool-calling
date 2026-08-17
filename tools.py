import requests
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ==========================================
# Ümumi HTTP sorğu konfiqurasiyası
# ==========================================
REQUEST_TIMEOUT = 5

# Xarici API-lər (açar tələb olunmur)
IP_WHO_URL = "https://ipwho.is/"
IP_API_FALLBACK_URL = "http://ip-api.com/json/"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def _safe_get_json(url: str, params: Optional[dict] = None) -> dict:
    """Xarici API-yə sorğu göndərir və JSON cavabını qaytarır.

    Şəbəkə və ya HTTP xətaları zamanı istifadəçiyə anlaşılan mesaj
    verən istisnalar (exceptions) qaldırır.

    Args:
        url (str): Sorğu göndəriləcək API endpoint.
        params (Optional[dict]): URL-ə əlavə olunacaq query parametrləri.

    Returns:
        dict: API-dən gələn JSON obyekti.

    Raises:
        ConnectionError: Şəbəkə/connection xətası olduqda.
        RuntimeError: API uğursuz HTTP status kodu qaytardıqda.
    """
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise ConnectionError(f"API sorğusu vaxt aşımına uğradı ({url}).") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"API-yə qoşulmaq mümkün olmadı ({url}).") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"API HTTP xətası qaytardı (status: {response.status_code})."
        ) from exc


def _geocode_city(city: str) -> tuple[str, float, float]:
    """Şəhər adını Open-Meteo Geocoding API ilə koordinatlara çevirir.

    Args:
        city (str): Axtarılacaq şəhər adı (məsələn: 'Paris').

    Returns:
        tuple[str, float, float]: (şəhər adı, enlik/latitude, uzunluq/longitude).

    Raises:
        ValueError: Şəhər tapılmadıqda.
    """
    data = _safe_get_json(
        OPEN_METEO_GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise ValueError(f"'{city}' şəhəri coğrafi kodlama (geocoding) nəticəsində tapılmadı.")
    first = results[0]
    return first["name"], float(first["latitude"]), float(first["longitude"])


def _detect_ip_location() -> tuple[str, float, float]:
    """İstifadəçinin IP ünvanına görə cari məkanı müəyyən edir.

    Əsas servis (ipwho.is) uğursuz olarsa ehtiyat servisə (ip-api.com)
    keçilir. Hər ikisi açar tələb etmir.

    Returns:
        tuple[str, float, float]: (şəhər adı, latitude, longitude).

    Raises:
        ConnectionError: Hər iki IP geolocation servisi əlçatan olmadıqda.
    """
    errors = []
    try:
        data = _safe_get_json(IP_WHO_URL)
        if data.get("success") is not False and data.get("city"):
            return str(data["city"]), float(data["latitude"]), float(data["longitude"])
        errors.append("ipwho.is cavabı etibarsızdır.")
    except (ConnectionError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))

    try:
        data = _safe_get_json(IP_API_FALLBACK_URL)
        if data.get("status") == "success" and data.get("city"):
            return str(data["city"]), float(data["lat"]), float(data["lon"])
        errors.append("ip-api.com cavabı etibarsızdır.")
    except (ConnectionError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))

    raise ConnectionError("Cari məkan IP geolocation servisləri ilə müəyyən edilə bilmədi. " + " | ".join(errors))


# ==========================================
# 1. TOOL: Məkan Məlumatının Alınması
# ==========================================
class LocationInput(BaseModel):
    city: Optional[str] = Field(
        default=None,
        description=(
            "Hava haqqında soruşulan şəhərin adı (məsələn: 'Paris'). "
            "Əgər istifadəçi 'burada', 'haradayam', 'cari yerləşdiyim yerdə' kimi "
            "ifadələr işlədibsə, bu parametr GÖNDƏRİLMƏMƏLİDİR."
        ),
    )


@tool(args_schema=LocationInput)
def get_current_location(city: Optional[str] = None) -> str:
    """İstifadəçinin cari məkanını (şəhər + koordinatlar) qaytarır.

    QAYDA:
    1. İstifadəçi sorğusunda konkret şəhər adı çəkilməyibsə (məsələn:
       'burada hava necədir?', 'haradayam?', 'hava mənim yerimdə necədir?')
       bu tool heç bir parametr olmadan çağırılmalıdır və IP ünvanına görə
       cari məkan müəyyən edilir.
    2. İstifadəçi konkret şəhər adı çəkirsə (məsələn: 'Parijdə hava necədir?')
       həmin şəhər Open-Meteo Geocoding API ilə koordinatlara çevrilir.

    Args:
        city (Optional[str]): Koordinatları alınacaq şəhər adı. İstifadəçinin
            cari məkanı tələb olunduqda buraxılır (None).

    Returns:
        str: 'Şəhər: <ad> | Enlik: <lat> | Uzunluq: <lon>' formatında
            məkan məlumatı. Növbəti addımda `get_weather_by_coordinates`
            tool-u üçün lazım olan koordinatları ehtiva edir.

    Raises:
        ValueError: Göstərilən şəhər geocoding nəticəsində tapılmadıqda.
        ConnectionError: Cari məkan müəyyən edilə bilmədikdə.
    """
    if city:
        resolved_city, latitude, longitude = _geocode_city(city)
    else:
        resolved_city, latitude, longitude = _detect_ip_location()

    return (
        f"Şəhər: {resolved_city} | Enlik: {latitude:.4f} | Uzunluq: {longitude:.4f}"
    )


# ==========================================
# 2. TOOL: Hava Məlumatının Alınması
# ==========================================
class WeatherInput(BaseModel):
    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Hava məlumatı alınacaq məkanın enliyi (latitude, -90 ilə 90 arası).",
    )
    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Hava məlumatı alınacaq məkanın uzunluğu (longitude, -180 ilə 180 arası).",
    )


@tool(args_schema=WeatherInput)
def get_weather_by_coordinates(latitude: float, longitude: float) -> str:
    """Verilmiş koordinatlar üçün cari hava temperaturunu Selsi (°C) ilə qaytarır.

    QAYDA:
    1. Koordinatlar məlum deyilsə, bu tool-u çağırmazdan öncə
       `get_current_location` istifadə edilməlidir.
    2. Bu tool yalnız hava məlumatını almaq üçündür; dərəcə çevrilməsi
       (Fahrenheit-ə) ETMİR. Həmin əməliyyat üçün `convert_celsius_to_fahrenheit`
       ayrıca çağırılmalıdır.
    3. İstifadəçi yalnız çevrilmə istəyirsə (məsələn: '30 dərəcə Selsi neçə
       Fahrenheit edir?') bu tool çağırılmamalıdır.

    Args:
        latitude (float): Məkanın enliyi (-90 ilə 90 arası).
        longitude (float): Məkanın uzunluğu (-180 ilə 180 arası).

    Returns:
        str: Selsi (°C) ilə cari temperatur və hava şəraitini əks etdirən
            məlumat sətri.

    Raises:
        ConnectionError: Weather API-yə qoşulmaq mümkün olmadıqda.
        RuntimeError: Weather API xəta qaytardıqda.
    """
    data = _safe_get_json(
        OPEN_METEO_WEATHER_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
        },
    )
    current = data.get("current") or {}
    temperature = current.get("temperature_2m")
    if temperature is None:
        raise RuntimeError("Weather API cavabında temperatur məlumatı tapılmadı.")

    weather_code = current.get("weather_code", "Bilinmir")
    wind_speed = current.get("wind_speed_10m", "Bilinmir")
    return (
        f"{data.get('timezone', 'Yerli')}: {temperature}°C, "
        f"weather_code={weather_code}, külək sürəti={wind_speed} km/saat"
    )


# ==========================================
# 3. TOOL: Temperatur Çevrilməsi
# ==========================================
class TemperatureConversionInput(BaseModel):
    celsius: float = Field(
        description="Fahrenheit-ə çevriləcək dərəcə dəyəri (Selsi ilə, məsələn: 22.0)."
    )


@tool(args_schema=TemperatureConversionInput)
def convert_celsius_to_fahrenheit(celsius: float) -> str:
    """Selsi (°C) dərəcəsini Fahrenheit (°F) vahidinə çevirir.

    QAYDA: Yalnız və yalnız istifadəçi xüsusi olaraq dərəcənin Fahrenheit-ə
    çevrilməsini tələb etdikdə çağırılmalıdır. Sadə hava sorğularında
    ('Bakıda hava necədir?') bu tool-u çağırmayın.

    Args:
        celsius (float): Selsi ilə dərəcə sayı.

    Returns:
        str: Fahrenheit ilə hesablanmış nəticə sətri.

    Raises:
        ValueError: celsius dəyəri etibarsız olduqda.
    """
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C dərəcə {fahrenheit:.1f}°F dərəcəyə bərabərdir."