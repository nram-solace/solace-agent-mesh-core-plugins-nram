# Solace Agent Mesh Geographic Information

A plugin that provides comprehensive geographic information services including location lookup, timezone data, and weather information.

## Features

- Convert city names to geographic coordinates
- Look up timezone information for cities
- Get current weather conditions and forecasts for locations

## Add a Geographic Location Agent to SAM

Add the plugin to your SAM instance:

```sh
solace-agent-mesh plugin add sam_geo_information --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-geo-information
```

To instantiate the agent, you can edit the SAM configuration file solace-agent-mesh.yaml:

```
  ...
  plugins:
  - name: sam_geo_information
    load_unspecified_files: false
    includes_gateway_interface: false
    load:
      agents: 
        - geo_information  # Add this line
      gateways: []
      overwrites: []
    from_url: 
      git+https://github.com/SolaceDev/solace-agent-mesh-core-plugins@ed/add-geo-info-agent#subdirectory=sam-geo-information
  ...
```

or this will create a new agent config file in your agent config directory:

```sh
solace-agent-mesh add agent geo_information --copy-from sam_geo_information
```

This will create a new config file in your agent config directory. Rename this file to the agent name you want to use.
Also update the following fields in the config file:
- **agent_name**
- **name (flow name)**
- **broker_subscriptions.topic**

## Environment Variables

The following environment variables are required:
- **SOLACE_BROKER_URL**
- **SOLACE_BROKER_USERNAME**
- **SOLACE_BROKER_PASSWORD**
- **SOLACE_BROKER_VPN**
- **SOLACE_AGENT_MESH_NAMESPACE**

For the GeoCode API:
- **GEOCODING_API_KEY**

For the Weather API (require for commercial use only):
- **WEATHER_API_KEY**

## Actions

### city_to_coordinates
Converts a city name to its geographic coordinates. If multiple matches are found, all possibilities will be returned.

### city_to_timezone
Converts a city name to its timezone information, including current UTC offset and DST status.

### get_weather
Gets current weather conditions and optional forecast for a location. Supports both metric and imperial units.


## GeoCode API

Currently, this agent uses geocode.maps.co as the GeoCode API. You will need to sign up for an API key at [https://geocode.maps.co/](https://geocode.maps.co/) if you want to use it with any substantial volume.

## Weather API

Currently, this agent uses Open Meteo as the weather API. You may use this for free as long as it is for non-commercial purposes. For commercial use, you will need to sign up for an API key at [https://open-meteo.com/](https://open-meteo.com/).