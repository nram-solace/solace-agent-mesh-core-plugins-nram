# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the Solace Agent Mesh core plugins - a collection of Python-based agent plugins that extend the Solace Agent Mesh framework. Each plugin is a self-contained package that provides specific capabilities like RAG document processing, SQL database querying, geo-information services, and more.

## Repository Structure

The repository follows a plugin-per-directory structure:
- Each plugin directory (e.g., `sam-rag`, `sam-sql-database`) is a complete Python package
- Each plugin contains its own `pyproject.toml`, README, configuration files, and source code
- All plugins follow the same structure: `src/agents/<plugin_name>/` for core logic, `configs/agents/` for templates, and `tests/` for testing

## Development Commands

### Plugin Development
```bash
# Install a plugin into your solace-agent-mesh project
solace-agent-mesh plugin add PLUGIN_NAME --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=PLUGIN_NAME

# Create an agent instance from a plugin template
solace-agent-mesh add agent my_agent --copy-from sam_plugin_name:template_name

# Build the agent mesh after adding plugins/agents  
solace-agent-mesh build

# Run the agent mesh
solace-agent-mesh run
```

### Testing
```bash
# For plugins with test runners (like sam-rag)
cd sam-rag && python tests/run_tests.py

# For basic plugin tests
cd PLUGIN_NAME && python -m pytest tests/
```

### Package Building
Each plugin uses Hatch for packaging:
```bash
cd PLUGIN_NAME
hatch build
```

## Plugin Architecture

### Core Components
- **Agent Component**: Main class implementing the agent logic (e.g., `RagAgentComponent`)
- **Actions**: Individual operations the agent can perform (in `actions/` directory)
- **Services**: Shared business logic and external integrations (in `services/` directory)
- **Configuration**: YAML-based configuration with environment variable templating

### Configuration Templating System
Plugins use a templating system for multi-instance deployment:
- Template variables: `{{SNAKE_CASE_NAME}}` becomes the instance name
- Environment variables: `{{SNAKE_UPPER_CASE_NAME}}` becomes the uppercase prefix
- Example: For instance `my_rag`, config uses `${MY_RAG_DATABASE_URL}`

### Plugin Registration
Each plugin has a `solace-agent-mesh-plugin.yaml` file defining:
- Plugin name and metadata
- Whether it includes gateway interfaces
- Build configuration

## Key Plugin Categories

### Document Processing (sam-rag)
- Comprehensive RAG pipeline with document scanning, preprocessing, embedding, and retrieval
- Supports multiple vector databases (Qdrant, Pinecone, PGVector, etc.)
- Cloud storage integration (S3, Google Drive, OneDrive)
- Extensive documentation in `docs/` directory

### Database Integration (sam-sql-database)
- Multi-database support (MySQL, PostgreSQL, MSSQL, SQLite)
- Natural language to SQL conversion
- CSV import capabilities
- Automatic schema detection

### External Services
- **sam-geo-information**: Weather and geocoding services
- **sam-bedrock-agent**: AWS Bedrock integration
- **sam-mcp-server**: Model Context Protocol server integration
- **sam-mermaid**: Diagram generation
- **sam-mongodb**: MongoDB query interface
- **sam-ml-scikit-learn**: Machine learning operations

### Event Processing (solace-event-mesh)
- Solace PubSub+ integration with request/response patterns
- Dynamic action creation based on configuration
- Gateway interfaces for event-driven architectures

## Testing Strategy

The repository includes a comprehensive testing checklist (`TESTING_CHECKLIST.md`) covering:
- Agent instantiation via `add agent` command
- Environment variable handling with new naming conventions
- Basic functionality tests for each agent type
- Single instance vs multi-instance testing
- Documentation verification

## Environment Configuration

Each plugin supports environment-based configuration:
- Required variables must be set for agent startup
- Optional variables have sensible defaults
- Naming convention: `${INSTANCE_NAME_SETTING}` (uppercase)
- Templates support both single-instance and multi-instance deployments

## Development Patterns

### Adding New Actions
1. Create action class in `src/agents/PLUGIN/actions/`
2. Implement required methods following existing patterns
3. Register action in agent component
4. Add configuration support in YAML templates

### Service Integration
1. Create service class in `src/agents/PLUGIN/services/`
2. Follow dependency injection patterns
3. Use configuration-driven initialization
4. Implement proper error handling and logging

### Plugin Creation
1. Follow existing directory structure
2. Create `pyproject.toml` with proper dependencies
3. Implement agent component following base patterns
4. Add configuration templates with environment variable support
5. Include comprehensive README and documentation