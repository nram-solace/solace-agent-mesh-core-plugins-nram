"""Action for converting city names to timezone information."""

from typing import Dict, Any
from timezonefinder import TimezoneFinder
import pytz
import yaml
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

from ..services.geocoding_service import MapsCoGeocodingService


class CityToTimezone(Action):
    """Convert city names to timezone information."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "city_to_timezone",
                "prompt_directive": (
                    "Convert a city name to its timezone information. "
                    "Returns timezone name, current offset from UTC, and whether DST is active."
                ),
                "params": [
                    {
                        "name": "city",
                        "desc": "Location to look up. Can be a city name (e.g., 'Paris'), city and country (e.g., 'Paris, France'), or full address. More specific inputs will return more precise results.",
                        "type": "string",
                        "required": True,
                    }
                ],
                "required_scopes": ["<agent_name>:city_to_timezone:execute"],
            },
            **kwargs
        )
        geocoding_api_key = kwargs.get("config_fn")("geocoding_api_key")
        self.geocoding_service = MapsCoGeocodingService(api_key=geocoding_api_key)
        self.timezone_finder = TimezoneFinder()

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the city to timezone conversion.
        
        Args:
            params: Must contain 'city' parameter
            meta: Optional metadata
            
        Returns:
            ActionResponse containing the timezone information or error information
        """
        try:
            city = params.get("city")
            if not city:
                raise ValueError("City parameter is required")

            locations = self.geocoding_service.geocode(city)
            if not locations:
                raise ValueError(f"No locations found for city: {city}")

            results = []
            for loc in locations:
                # Get timezone name for the coordinates
                timezone_str = self.timezone_finder.timezone_at(
                    lat=loc.latitude, 
                    lng=loc.longitude
                )
                if not timezone_str:
                    continue

                # Get timezone object and current time info
                timezone = pytz.timezone(timezone_str)
                now = pytz.datetime.datetime.now(timezone)
                
                result = {
                    "location": loc.display_name,
                    "timezone": timezone_str,
                    "utc_offset": now.strftime("%z"),
                    "dst_active": bool(now.dst()),
                    "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z")
                }
                results.append(result)

            if not results:
                raise ValueError(f"Could not determine timezone for any matching location: {city}")

            if len(results) == 1:
                message = f"Found timezone information for {city}:\n\n{yaml.dump(results)}"
            else:
                message = f"Found timezone information for {len(results)} possible matches for {city}:\n\n{yaml.dump(results)}"

            return ActionResponse(message=message)

        except Exception as e:
            return ActionResponse(
                message=f"Error looking up timezone: {str(e)}",
                error_info=ErrorInfo(str(e))
            )
