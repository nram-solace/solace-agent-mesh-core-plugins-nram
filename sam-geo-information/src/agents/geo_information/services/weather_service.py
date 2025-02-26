"""Service for handling weather data retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import requests


class Units(Enum):
    """Supported unit systems."""
    METRIC = "metric"
    IMPERIAL = "imperial"


@dataclass
class WeatherData:
    """Weather data container."""
    temperature: float
    feels_like: float
    humidity: float
    wind_speed: float
    precipitation: float
    cloud_cover: int
    pressure: float
    units: Units
    description: str
    timestamp: datetime


class WeatherService(ABC):
    """Abstract base class for weather services."""

    @abstractmethod
    def get_current_weather(self, latitude: float, longitude: float, units: Units = Units.METRIC) -> WeatherData:
        """Get current weather for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            units: Unit system to use
            
        Returns:
            WeatherData object with current conditions
            
        Raises:
            ValueError: If weather data cannot be retrieved
        """
        pass

    @abstractmethod
    def get_forecast(self, latitude: float, longitude: float, days: int = 7, 
                    units: Units = Units.METRIC) -> List[WeatherData]:
        """Get weather forecast for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of days to forecast
            units: Unit system to use
            
        Returns:
            List of WeatherData objects for each day
            
        Raises:
            ValueError: If forecast cannot be retrieved
        """
        pass


class OpenMeteoWeatherService(WeatherService):
    """Weather service implementation using Open-Meteo API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Open-Meteo weather service.
        
        Args:
            api_key: Optional API key for the weather service
        """
        self.base_url = "https://api.open-meteo.com/v1"
        self.api_key = api_key

    def get_current_weather(self, latitude: float, longitude: float, units: Units = Units.METRIC) -> WeatherData:
        """Get current weather from Open-Meteo.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            units: Unit system to use
            
        Returns:
            WeatherData object with current conditions
            
        Raises:
            ValueError: If weather data cannot be retrieved
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "apikey": self.api_key if self.api_key else None,
                "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                           "precipitation", "cloud_cover", "pressure_msl", "wind_speed_10m"],
                "wind_speed_unit": "mph" if units == Units.IMPERIAL else "kmh",
                "temperature_unit": "fahrenheit" if units == Units.IMPERIAL else "celsius",
                "precipitation_unit": "inch" if units == Units.IMPERIAL else "mm"
            }

            response = requests.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            if not current:
                raise ValueError("No current weather data available")

            return WeatherData(
                temperature=current["temperature_2m"],
                feels_like=current["apparent_temperature"],
                humidity=current["relative_humidity_2m"],
                wind_speed=current["wind_speed_10m"],
                precipitation=current["precipitation"],
                cloud_cover=current["cloud_cover"],
                pressure=current["pressure_msl"],
                units=units,
                description=self._generate_description(current),
                timestamp=datetime.fromisoformat(current["time"])
            )

        except requests.RequestException as e:
            raise ValueError(f"Weather request failed: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error parsing weather response: {str(e)}") from e

    def get_forecast(self, latitude: float, longitude: float, days: int = 7,
                    units: Units = Units.METRIC) -> List[WeatherData]:
        """Get weather forecast from Open-Meteo.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of days to forecast
            units: Unit system to use
            
        Returns:
            List of WeatherData objects for each day
            
        Raises:
            ValueError: If forecast cannot be retrieved
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
                         "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_max"],
                "wind_speed_unit": "mph" if units == Units.IMPERIAL else "kmh",
                "temperature_unit": "fahrenheit" if units == Units.IMPERIAL else "celsius",
                "precipitation_unit": "inch" if units == Units.IMPERIAL else "mm",
                "forecast_days": min(days, 16)  # API limit is 16 days
            }

            response = requests.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            if not daily:
                raise ValueError("No forecast data available")

            forecast = []
            for i in range(len(daily["time"])):
                forecast.append(WeatherData(
                    temperature=(daily["temperature_2m_max"][i] + daily["temperature_2m_min"][i]) / 2,
                    feels_like=daily["apparent_temperature_max"][i],
                    humidity=daily["relative_humidity_2m_max"][i],
                    wind_speed=daily["wind_speed_10m_max"][i],
                    precipitation=daily["precipitation_sum"][i],
                    cloud_cover=0,  # Not available in daily forecast
                    pressure=0,  # Not available in daily forecast
                    units=units,
                    description=self._generate_daily_description(daily, i),
                    timestamp=datetime.fromisoformat(daily["time"][i])
                ))

            return forecast

        except requests.RequestException as e:
            raise ValueError(f"Forecast request failed: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error parsing forecast response: {str(e)}") from e

    def _generate_description(self, current: Dict) -> str:
        """Generate a human-readable weather description."""
        temp_unit = "°F" if current.get("temperature_unit") == "fahrenheit" else "°C"
        speed_unit = "mph" if current.get("wind_speed_unit") == "mph" else "km/h"
        
        return (
            f"Temperature: {current['temperature_2m']}{temp_unit}, "
            f"Feels like: {current['apparent_temperature']}{temp_unit}, "
            f"Wind: {current['wind_speed_10m']}{speed_unit}, "
            f"Humidity: {current['relative_humidity_2m']}%"
        )

    def _generate_daily_description(self, daily: Dict, index: int) -> str:
        """Generate a human-readable daily forecast description."""
        temp_unit = "°F" if daily.get("temperature_unit") == "fahrenheit" else "°C"
        
        return (
            f"High: {daily['temperature_2m_max'][index]}{temp_unit}, "
            f"Low: {daily['temperature_2m_min'][index]}{temp_unit}, "
            f"Precipitation: {daily['precipitation_sum'][index]}mm"
        )
