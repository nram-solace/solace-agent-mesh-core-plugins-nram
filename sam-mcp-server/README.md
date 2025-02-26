# MCP

This adds both an agent to talk to MCP servers and a gateway for SAM to act as an MCP server

## Add a MCP server to SAM

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add sam_mcp_server --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-mcp-server
```

### One MCP server

To use **one MCP server** you can just import the pre-made agent, by updating your `solace-agent-mesh.yaml` file

```json
...
  plugins:

  ...

  - name: sam_mcp_server
    load_unspecified_files: false
    includes_gateway_interface: true
    load:
      agents:
        - mcp_server
      gateways: []
      overwrites: []

   ...
```

And provide the following ENV variables:
- **MCP_SERVER_NAME**
- **MCP_SERVER_COMMAND**


### Multiple MCP servers

To instantiate the agent, (eg if you're using multiple agents) you can use the following code:

```sh
solace-agent-mesh add agent mcp_server --copy-from sam_mcp_server
```

This will create a new config file in your agent config directory. Rename this file to your MCP server name.
You can also rename or hard-code the following ENV variables:
- **MCP_SERVER_NAME**
- **MCP_SERVER_COMMAND**

**Example ENV values**

```
MCP_SERVER_NAME=filesystem
MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-filesystem /Path/To/Allow/Access
```

or

```sh
MCP_SERVER_NAME=server-everything
MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-everything
```
