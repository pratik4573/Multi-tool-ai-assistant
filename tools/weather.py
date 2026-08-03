"""Weather lookup tool (uses OpenWeatherMap if an API key is configured)."""

from __future__ import annotations

import requests

import config

SPEC = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Mumbai'."}
        },
        "required": ["city"],
    },
}


def run(args: dict) -> str:
    city = str(args.get("city", "")).strip()
    if not city:
        return "Error: no city provided."

    if not config.WEATHER_API_KEY:
        return (
            "Weather tool is not configured. Set OPENWEATHER_API_KEY in your "
            ".env to enable live weather lookups."
        )

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": config.WEATHER_API_KEY, "units": "metric"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return (
            f"{city}: {description}, {temp}°C (feels like {feels_like}°C)."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not fetch weather for '{city}' ({exc})."
