# Solace Agent Mesh RAG

A document-ingesting agent that monitors specified directories, keeping stored documents up to date in a vector database for Retrieval-Augmented Generation (RAG) queries.

## Add a RAG Agent to SAM

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add sam_rag --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-rag
```

To instantiate the agent, you can use the following code: (you can repeat this step to connect to multiple collections/databases)

```sh
solace-agent-mesh add agent rag --copy-from sam_rag
```

This will create a new config file in your agent config directory. Rename this file to the agent name you want to use. 
Also update the following fields in the config file:
- **agent_name**
- <add more>
