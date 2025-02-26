"""Geographic location agent component for handling location-based operations."""

import copy
from typing import Dict, Any

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .actions.city_to_coordinates import CityToCoordinates
from .actions.city_to_timezone import CityToTimezone
from .actions.get_weather import GetWeather


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "geo_information",
        "class_name": "GeoInformationAgentComponent",
        "description": "Geographic information agent for handling location, timezone, and weather operations for locations in the world.",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this geographic information agent",
                "type": "string",
            },
            {
                "name": "geocoding_api_key",
                "required": False,
                "description": "API key for the geocoding service (maps.co)",
                "type": "string",
            },
            {
                "name": "weather_api_key",
                "required": False,
                "description": "API key for the weather service (open-meteo)",
                "type": "string",
            }
        ],
    }
)


class GeoInformationAgentComponent(BaseAgentComponent):
    """Component for handling geographic information operations including location, timezone, and weather."""

    info = info
    actions = [CityToCoordinates, CityToTimezone, GetWeather]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize this agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.
        """
        module_info = module_info or info
        super().__init__(module_info, **kwargs)

        self.agent_name = self.get_config("agent_name")
        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        module_info["agent_name"] = self.agent_name

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return {
            "agent_name": self.agent_name,
            "description": (
                "This agent provides comprehensive geographic information services including:\n"
                "- Location Services: Converting city names to precise coordinates\n"
                "- Timezone Information: Looking up timezone data, UTC offsets, and DST status\n"
                "- Weather Services: Current conditions, forecasts, and historical weather data\n\n"
                "All of these services work together to provide detailed geographic insights for any location."
            ),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }
