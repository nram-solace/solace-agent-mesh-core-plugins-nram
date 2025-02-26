"""Service for handling geocoding operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import requests


@dataclass
class GeoLocation:
    """Represents a geographic location."""
    latitude: float
    longitude: float
    display_name: str
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None


class GeocodingService(ABC):
    """Abstract base class for geocoding services."""

    @abstractmethod
    def geocode(self, location: str) -> List[GeoLocation]:
        """Convert a location string to geographic coordinates.
        
        Args:
            location: Location string to geocode
            
        Returns:
            List of GeoLocation objects representing possible matches
            
        Raises:
            ValueError: If geocoding fails
        """
        pass


class MapsCoGeocodingService(GeocodingService):
    """Geocoding service implementation using geocode.maps.co."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Maps.co geocoding service.
        
        Args:
            api_key: Optional API key for the geocoding service
        """
        self.base_url = "https://geocode.maps.co/search"
        self.api_key = api_key

    def geocode(self, location: str) -> List[GeoLocation]:
        """Convert a location string to geographic coordinates using Maps.co.
        
        Args:
            location: Location string to geocode
            
        Returns:
            List of GeoLocation objects representing possible matches
            
        Raises:
            ValueError: If geocoding fails or returns no results
        """
        try:
            params = {"q": location}
            if self.api_key:
                params["api_key"] = self.api_key
                
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            results = response.json()

            if not results:
                raise ValueError(f"No results found for location: {location}")

            locations = []
            for result in results:
                locations.append(GeoLocation(
                    latitude=float(result["lat"]),
                    longitude=float(result["lon"]),
                    display_name=result["display_name"],
                    # Parse display_name to extract components
                    country=result.get("address", {}).get("country"),
                    state=result.get("address", {}).get("state"),
                    city=result.get("address", {}).get("city")
                ))

            return locations

        except requests.RequestException as e:
            raise ValueError(f"Geocoding request failed: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error parsing geocoding response: {str(e)}") from e
