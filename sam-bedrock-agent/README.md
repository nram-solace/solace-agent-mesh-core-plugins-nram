# SAM-BEDROCK-AGENT

This plugin allows you to import one or multiple Amazon bedrock agents or flows as action to be used in your SAM project.



## Add the Plugin

If you haven't already, add the plugin to your SAM instance:
```sh
solace-agent-mesh plugin add solace-event-mesh --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-bedrock-agent
```

## Using the Bedrock Agent

The Event Mesh Agent allows you to define custom actions within SAM that correspond to request/reply interactions over the Solace Event Mesh.

### Instantiate the Agent

Use the `solace-agent-mesh add agent` command to create a configuration file for your specific Bedrock agent instance.

```sh
solace-agent-mesh add agent <new_agent_name> --copy-from sam_bedrock_agent:bedrock_agent
```

- Replace `<new_agent_name>` with your chosen name.

This creates `<new_agent_name>.yaml` in `configs/agents/`. You will configure the agent by editing this new file.

### Configuration (`configs/agents/solace_event_mesh.yaml`)

The core of the agent's configuration are the `amazon_bedrock_runtime_config` and `bedrock_agents` and `bedrock_flows` list within the `component_config` section. Each item in the list defines an action that the agent will expose.

```yaml
# ... other config ...
flows:
  - name: {{SNAKE_CASE_NAME}}_action_request_processor
    components:
      # ... broker_input ...
      - component_name: action_request_processor
        component_module: {{MODULE_DIRECTORY}}.agents.bedrock_agent.bedrock_agent_agent_component
        component_config:
          agent_name: {{SNAKE_CASE_NAME}} # Your agent name as seen by SAM
          # ... other component settings ...
          amazon_bedrock_runtime_config:
            # AWS Endpoint URL - Optional
            endpoint_url:
            # AWS S3 configuration - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html
            # The following object is passed as kwargs to boto3.session.Session
            boto3_config:
              # AWS region name
              region_name: "us-east-1"
              # AWS credentials
              aws_access_key_id: # You can also use profiles instead, check boto3 documentation
              aws_secret_access_key:

          bedrock_agents: # Optional, but at least one bedrock_agent or bedrock_flow must be provided
            # EXAMPLE of a bedrock agent, Add as many needed
            - name: invoke_agent # The name of the action
              description: "Invoke the bedrock agent" # Description of the Amazon bedrock agent
              param_description: "Prompt to send to the action." # [Optional] Description of the parameter to be pass to the action, 
              bedrock_agent_id: "FAKE_AGENT_ID" # The ID of the Amazon bedrock agent
              bedrock_agent_alias_id: "FAKE_AGENT_ALIAS_ID" # The alias ID of the Amazon bedrock agent
              # required_scope: "<agent_name>:my_request_action:write" # Optional scope override
            # --- Add more agents ---

          bedrock_flows: # Optional, but at least one bedrock_agent or bedrock_flow must be provided
            # EXAMPLE of a bedrock flow, Add as many needed
            - name: invoke_flow # The name of the action
              description: "Invoke the bedrock flow" # Description of the Amazon bedrock flow
              param_description: "Prompt to send to the flow." # [Optional] Description of the parameter to be pass to the action, 
              bedrock_flow_id: "FAKE_FLOW_ID" # The ID of the Amazon bedrock flow
              bedrock_flow_alias_id: "FAKE_FLOW_ALIAS_ID" # The alias ID of the Amazon bedrock flow
              # required_scope: "<agent_name>:my_request_action:write" # Optional scope override
            # --- Add more flows ---
      # ... broker_request_response, broker_output ...
```

### Environment Variables

Only the standard Solace connection variables are required for the agent itself:
- **SOLACE_BROKER_URL**
- **SOLACE_BROKER_USERNAME**
- **SOLACE_BROKER_PASSWORD**
- **SOLACE_BROKER_VPN**
- **SOLACE_AGENT_MESH_NAMESPACE**

Any environment variable required for boto3 to create an AWS session
