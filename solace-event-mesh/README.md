# SOLACE-EVENT-MESH

This plugin adds a Gateway and Agent to connect to Solace's Event Mesh and Event Brokers

- [Event Mesh Gateway](./docs/event_mesh_gateway.md): Enables bidirectional communication with Solace Event Mesh.

## Create gateway instance

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add solace-event-mesh --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=solace-event-mesh
```

Create a new gateway with `solace-event-mesh` interface.


```sh
solace-agent-mesh add gateway my-event-mesh --interface solace-event-mesh
```

- replace `my-event-mesh` with your desired gateway name