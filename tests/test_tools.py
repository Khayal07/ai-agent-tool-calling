import requests
import pytest
from pydantic import ValidationError

import tools
from tools import (
    get_current_location,
    get_weather_by_coordinates,
    convert_celsius_to_fahrenheit,
)


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def _patch_requests(monkeypatch, responses):
    calls = {"urls": []}

    def fake_get(url, params=None, timeout=None):
        calls["urls"].append(url)
        for handler in responses:
            if callable(handler):
                result = handler(url)
                if result is not None:
                    return result
        return responses[0](url)

    monkeypatch.setattr(tools.requests, "get", fake_get)
    return calls


def test_convert_celsius_to_fahrenheit():
    result = convert_celsius_to_fahrenheit.invoke({"celsius": 22})
    assert result == "22.0°C dərəcə 71.6°F dərəcəyə bərabərdir."


def test_convert_celsius_to_fahrenheit_negative():
    result = convert_celsius_to_fahrenheit.invoke({"celsius": -40})
    assert "-40.0" in result and "-40.0°F" in result


def test_convert_celsius_to_fahrenheit_invalid_args():
    with pytest.raises(ValidationError):
        convert_celsius_to_fahrenheit.invoke({"celsius": "invalid"})


def test_get_current_location_with_city_uses_geocoding(monkeypatch):
    calls = _patch_requests(
        monkeypatch,
        [
            lambda url: FakeResponse(
                {
                    "results": [
                        {"name": "Paris", "latitude": 48.85341, "longitude": 2.3488}
                    ]
                }
            )
        ],
    )
    result = get_current_location.invoke({"city": "Paris"})
    assert "Paris" in result and "48.8534" in result and "2.3488" in result
    assert tools.OPEN_METEO_GEOCODING_URL in calls["urls"]


def test_get_current_location_without_city_uses_ip(monkeypatch):
    calls = _patch_requests(
        monkeypatch,
        [
            lambda url: FakeResponse(
                {"city": "Baku", "latitude": 40.3771, "longitude": 49.8875}
            )
        ],
    )
    result = get_current_location.invoke({})
    assert "Baku" in result and "40.3771" in result and "49.8875" in result
    assert tools.IP_WHO_URL in calls["urls"]


def test_get_current_location_falls_back_to_secondary_ip_service(monkeypatch):
    seen = []

    def handler(url):
        seen.append(url)
        if url == tools.IP_WHO_URL:
            raise requests.exceptions.ConnectionError("offline")
        return FakeResponse(
            {"status": "success", "city": "London", "lat": 51.5074, "lon": -0.1278}
        )

    _patch_requests(monkeypatch, [handler])
    result = get_current_location.invoke({})
    assert "London" in result and "51.5074" in result
    assert tools.IP_API_FALLBACK_URL in seen


def test_get_current_location_unknown_city_raises_value_error(monkeypatch):
    _patch_requests(monkeypatch, [lambda url: FakeResponse({"results": []})])
    with pytest.raises(ValueError):
        get_current_location.invoke({"city": "NONEIX"})


def test_get_weather_by_coordinates_parses_response(monkeypatch):
    _patch_requests(
        monkeypatch,
        [
            lambda url: FakeResponse(
                {
                    "timezone": "GMT",
                    "current": {
                        "temperature_2m": 26.1,
                        "weather_code": 2,
                        "wind_speed_10m": 8.0,
                    },
                }
            )
        ],
    )
    result = get_weather_by_coordinates.invoke(
        {"latitude": 48.8534, "longitude": 2.3488}
    )
    assert "26.1°C" in result and "weather_code=2" in result


def test_get_weather_by_coordinates_invalid_lat_raises_validation_error():
    with pytest.raises(ValidationError):
        get_weather_by_coordinates.invoke({"latitude": 95, "longitude": 2.3})


def test_weather_api_connection_error_raises(monkeypatch):
    def handler(url):
        raise requests.exceptions.Timeout("timed out")

    _patch_requests(monkeypatch, [handler])
    with pytest.raises(ConnectionError):
        get_weather_by_coordinates.invoke({"latitude": 40.37, "longitude": 49.89})


def test_weather_api_http_error_raises_runtime(monkeypatch):
    _patch_requests(monkeypatch, [lambda url: FakeResponse({}, status_code=500)])
    with pytest.raises(RuntimeError):
        get_weather_by_coordinates.invoke({"latitude": 40.37, "longitude": 49.89})


def test_weather_api_missing_temperature_raises_runtime(monkeypatch):
    _patch_requests(monkeypatch, [lambda url: FakeResponse({"current": {}})])
    with pytest.raises(RuntimeError):
        get_weather_by_coordinates.invoke({"latitude": 40.37, "longitude": 49.89})
