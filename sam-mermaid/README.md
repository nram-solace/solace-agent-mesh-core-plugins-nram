# Solace Agent Mesh MERMAID

A plugin used to generate visualizations using Mermaid.js

## Add a Mermaid Agent to SAM

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add sam_mermaid --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-mermaid
```

To use the Mermaid agent, update your `solace-agent-mesh.yaml` file

```
...
  plugins:

  ...

  - name: sam_mermaid
    load_unspecified_files: false
    includes_gateway_interface: false
    load:
      agents:
        - mermaid
      gateways: []
      overwrites: []

   ...
```

**Note for Enterprise Users:** Ensure you update your authorization configuration before using the agent.

Set the `MERMAID_SERVER_URL` environment variable to point to your [mermaid-server](https://github.com/TomWright/mermaid-server) instance.
