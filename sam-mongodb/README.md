# Solace Agent Mesh MONGODB

A plugin the provides mongodb agent to perform complex queries

## Add a MongoDB Agent to SAM

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add sam_mongodb --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-mongodb
```

To instantiate the agent, you can use the following code: (you can repeat this step to connect to multiple collections/databases)

```sh
solace-agent-mesh add agent mongodb --copy-from sam_mongodb
```

This will create a new config file in your agent config directory. Rename this file to the agent name you want to use. 
Also update the following fields in the config file:
- **agent_name**
- **database_purpose**
- **data_description**
- **auto_detect_schema**
- **name (flow name)**
- **broker_subscriptions.topic**


And provide the appropriate Database connection environment variables for:
- **MONGODB_HOST**
- **MONGODB_PORT**
- **MONGODB_USER**
- **MONGODB_PASSWORD**
- **MONGODB_DB**
- **MONGODB_COLLECTION**