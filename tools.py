"""
tools.py - External world interface for Chronos agent.

Handles weather data fetching with:
- Simple in-memory caching
- Simulation mode for reliable demos
- Graceful fallbacks on API failures
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import httpx

from models import WeatherCondition


# Simple in-memory cache: {(location, date): (WeatherCondition, timestamp)}
_weather_cache: dict[tuple[str, str], tuple[WeatherCondition, datetime]] = {}
CACHE_TTL_MINUTES = 30


def _is_cache_valid(cache_time: datetime) -> bool:
    """Check if cached data is still fresh."""
    return datetime.now() - cache_time < timedelta(minutes=CACHE_TTL_MINUTES)


def _get_cached_weather(location: str, date: str) -> Optional[WeatherCondition]:
    """Retrieve weather from cache if valid."""
    key = (location.lower(), date)
    if key in _weather_cache:
        weather, cache_time = _weather_cache[key]
        if _is_cache_valid(cache_time):
            return weather
        # Expired - remove from cache
        del _weather_cache[key]
    return None


def _store_cached_weather(location: str, date: str, weather: WeatherCondition) -> None:
    """Store weather data in cache."""
    key = (location.lower(), date)
    _weather_cache[key] = (weather, datetime.now())


def generate_simulated_weather(location: str, date: str) -> WeatherCondition:
    """
    Generate realistic simulated weather for demos.
    Provides consistent results for same location/date within a session.
    """
    # Use location + date as seed for reproducible "random" weather
    seed_value = hash(f"{location.lower()}_{date}") % 10000
    random.seed(seed_value)
    
    conditions = ["sunny", "partly cloudy", "cloudy", "light rain", "rainy", "thunderstorms"]
    weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Bias toward good weather for demos
    
    condition = random.choices(conditions, weights=weights)[0]
    
    # Temperature based on condition
    base_temp = random.uniform(15, 28)
    if "rain" in condition or condition == "thunderstorms":
        base_temp -= random.uniform(3, 8)
    
    # Precipitation chance based on condition
    precip_map = {
        "sunny": random.randint(0, 10),
        "partly cloudy": random.randint(5, 25),
        "cloudy": random.randint(15, 40),
        "light rain": random.randint(50, 70),
        "rainy": random.randint(70, 90),
        "thunderstorms": random.randint(80, 100)
    }
    
    return WeatherCondition(
        temperature_celsius=round(base_temp, 1),
        condition=condition,
        precipitation_chance=precip_map[condition],
        wind_speed_kmh=round(random.uniform(5, 25), 1),
        humidity_percent=random.randint(40, 85),
        forecast_date=date,
        location=location,
        is_simulated=True
    )


async def fetch_weather_from_api(location: str, date: str) -> Optional[WeatherCondition]:
    """
    Fetch weather from wttr.in API (free, no key required).
    
    wttr.in provides simple weather data via a REST API.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # wttr.in JSON endpoint
            url = f"https://wttr.in/{location}?format=j1"
            resp = await client.get(url, headers={"User-Agent": "Chronos-Agent/1.0"})
            resp.raise_for_status()
            data = resp.json()
            
            # Extract current condition and forecast
            current = data.get("current_condition", [{}])[0]
            weather_list = data.get("weather", [])
            
            # Find the forecast for the requested date or use current
            forecast = None
            for w in weather_list:
                if w.get("date") == date:
                    forecast = w
                    break
            
            # Fall back to first forecast day if exact date not found
            if not forecast and weather_list:
                forecast = weather_list[0]
            
            if not forecast:
                return None
            
            # Extract temperature (average of max and min)
            temp_max = float(forecast.get("maxtempC", 20))
            temp_min = float(forecast.get("mintempC", 15))
            avg_temp = (temp_max + temp_min) / 2
            
            # Get hourly data for precipitation and wind (use midday)
            hourly = forecast.get("hourly", [])
            midday = hourly[4] if len(hourly) > 4 else hourly[0] if hourly else {}
            
            # Weather description
            condition = current.get("weatherDesc", [{}])[0].get("value", "partly cloudy").lower()
            
            # Precipitation chance
            precip_chance = int(midday.get("chanceofrain", 0))
            
            # Wind speed
            wind_speed = float(midday.get("windspeedKmph", 10))
            
            # Humidity
            humidity = int(midday.get("humidity", 65))
            
            # Resolved location name
            nearest_area = data.get("nearest_area", [{}])[0]
            resolved_name = nearest_area.get("areaName", [{}])[0].get("value", location)
            
            return WeatherCondition(
                temperature_celsius=round(avg_temp, 1),
                condition=condition,
                precipitation_chance=precip_chance,
                wind_speed_kmh=round(wind_speed, 1),
                humidity_percent=humidity,
                forecast_date=date,
                location=resolved_name,
                is_simulated=False
            )
            
    except Exception as e:
        # Log error but don't crash - return None to trigger fallback
        print(f"[Weather API Error] {type(e).__name__}: {e}")
        return None


async def get_weather(
    location: str,
    date: str,
    use_simulation: bool = False
) -> WeatherCondition:
    """
    Main weather access function with caching and fallback.
    
    Priority:
    1. Return cached data if available and fresh
    2. If simulation mode, generate simulated weather
    3. Try real API
    4. Fall back to simulation on API failure
    
    This function NEVER raises exceptions - always returns data.
    """
    # Check cache first
    cached = _get_cached_weather(location, date)
    if cached:
        return cached
    
    weather: WeatherCondition
    
    if use_simulation:
        # Simulation mode for demos
        weather = generate_simulated_weather(location, date)
    else:
        # Try real API
        api_result = await fetch_weather_from_api(location, date)
        if api_result:
            weather = api_result
        else:
            # API failed - fall back to simulation
            weather = generate_simulated_weather(location, date)
            # Mark it as simulated since API failed
            weather.is_simulated = True
    
    # Cache the result
    _store_cached_weather(location, date, weather)
    
    return weather


def clear_weather_cache() -> None:
    """Clear the weather cache (useful for testing)."""
    _weather_cache.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Tool interface for PydanticAI agent
# ──────────────────────────────────────────────────────────────────────────────

async def weather_tool(location: str, date: str, simulation_mode: bool = False) -> dict:
    """
    PydanticAI-compatible tool for weather lookup.
    
    Returns a dictionary that the agent can use for reasoning.
    """
    weather = await get_weather(location, date, use_simulation=simulation_mode)
    return weather.model_dump()
