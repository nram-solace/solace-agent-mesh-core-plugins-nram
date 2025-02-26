"""Action for retrieving weather information for a location."""

from typing import Dict, Any
import yaml
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

from ..services.geocoding_service import MapsCoGeocodingService
from ..services.weather_service import OpenMeteoWeatherService, Units


class GetWeather(Action):
    """Get weather information for a location."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "get_weather",
                "prompt_directive": (
                    "Get weather information for a location. "
                    "Can return current conditions and forecast in metric or imperial units."
                ),
                "params": [
                    {
                        "name": "location",
                        "desc": "Name of the location to get weather for",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "units",
                        "desc": "Unit system to use (metric or imperial)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "forecast_days",
                        "desc": "Number of days to forecast (0 for current only, max 16)",
                        "type": "integer",
                        "required": False,
                    }
                ],
                "required_scopes": ["<agent_name>:get_weather:execute"],
            },
            **kwargs
        )
        geocoding_api_key = kwargs.get("config_fn")("geocoding_api_key")
        weather_api_key = kwargs.get("config_fn")("weather_api_key")
        self.geocoding_service = MapsCoGeocodingService(api_key=geocoding_api_key)
        self.weather_service = OpenMeteoWeatherService(api_key=weather_api_key)

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the weather information retrieval.
        
        Args:
            params: Must contain 'location' parameter
            meta: Optional metadata
            
        Returns:
            ActionResponse containing the weather information or error information
        """
        try:
            location = params.get("location")
            if not location:
                raise ValueError("Location parameter is required")

            units_str = params.get("units", "metric").lower()
            if units_str not in ("metric", "imperial"):
                raise ValueError("Units must be either 'metric' or 'imperial'")
            units = Units.METRIC if units_str == "metric" else Units.IMPERIAL

            forecast_days = int(params.get("forecast_days", 0))
            if forecast_days < 0 or forecast_days > 16:
                raise ValueError("Forecast days must be between 0 and 16")

            # Get coordinates for the location
            locations = self.geocoding_service.geocode(location)
            if not locations:
                raise ValueError(f"No locations found for: {location}")

            # Use the first location match
            loc = locations[0]
            
            # Get current weather
            current = self.weather_service.get_current_weather(
                latitude=loc.latitude,
                longitude=loc.longitude,
                units=units
            )

            result = {
                "location": loc.display_name,
                "current": {
                    "temperature": current.temperature,
                    "feels_like": current.feels_like,
                    "humidity": current.humidity,
                    "wind_speed": current.wind_speed,
                    "precipitation": current.precipitation,
                    "cloud_cover": current.cloud_cover,
                    "pressure": current.pressure,
                    "description": current.description,
                    "timestamp": current.timestamp.isoformat()
                },
                "units": units.value
            }

            # Get forecast if requested
            if forecast_days > 0:
                forecast = self.weather_service.get_forecast(
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                    days=forecast_days,
                    units=units
                )
                
                result["forecast"] = [{
                    "temperature": day.temperature,
                    "feels_like": day.feels_like,
                    "humidity": day.humidity,
                    "wind_speed": day.wind_speed,
                    "precipitation": day.precipitation,
                    "description": day.description,
                    "timestamp": day.timestamp.isoformat()
                } for day in forecast]

            return ActionResponse(
                message=f"Weather information for {loc.display_name}:\n\n{yaml.dump(result)}"
            )

        except Exception as e:
            return ActionResponse(
                message=f"Error getting weather information: {str(e)}",
                error_info=ErrorInfo(str(e))
            )
