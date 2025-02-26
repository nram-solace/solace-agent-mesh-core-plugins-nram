"""Action for converting city names to geographic coordinates."""

from typing import Dict, Any
import yaml
from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo

from ..services.geocoding_service import MapsCoGeocodingService


class CityToCoordinates(Action):
    """Convert city names to geographic coordinates."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "city_to_coordinates",
                "prompt_directive": (
                    "Convert a city name to its geographic coordinates. "
                    "If multiple matches are found, all possibilities will be returned."
                ),
                "params": [
                    {
                        "name": "city",
                        "desc": "Location to look up. Can be a city name (e.g., 'Paris'), city and country (e.g., 'Paris, France'), or full address. More specific inputs will return more precise results.",
                        "type": "string",
                        "required": True,
                    }
                ],
                "required_scopes": ["<agent_name>:city_to_coordinates:execute"],
            },
            **kwargs
        )
        geocoding_api_key = kwargs.get("config_fn")("geocoding_api_key")
        self.geocoding_service = MapsCoGeocodingService(api_key=geocoding_api_key)

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the city to coordinates conversion.
        
        Args:
            params: Must contain 'city' parameter
            meta: Optional metadata
            
        Returns:
            ActionResponse containing the coordinates or error information
        """
        try:
            city = params.get("city")
            if not city:
                raise ValueError("City parameter is required")

            locations = self.geocoding_service.geocode(city)
            
            # Format the results
            results = []
            for loc in locations:
                result = {
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "display_name": loc.display_name
                }
                if loc.country:
                    result["country"] = loc.country
                if loc.state:
                    result["state"] = loc.state
                if loc.city:
                    result["city"] = loc.city
                results.append(result)

            if len(results) == 1:
                message = f"Found coordinates for {city}:\n\n{yaml.dump(results)}"
            else:
                message = f"Found {len(results)} possible matches for {city}:\n\n{yaml.dump(results)}"

            return ActionResponse(message=message)

        except Exception as e:
            return ActionResponse(
                message=f"Error looking up coordinates: {str(e)}",
                error_info=ErrorInfo(str(e))
            )
