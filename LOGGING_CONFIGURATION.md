# Logging Configuration for Solace Agent Mesh

## Overview

This document explains the logging configuration for different agents in the Solace Agent Mesh system and how to ensure each agent logs to its own specific log file.

## Current Logging Configuration

### ✅ Agents with Agent-Specific Log Files

The following agents now have their own dedicated log files:

1. **SQL Database Agent**: `logs/sam-sql-database.log`
2. **ML Pandas Agent**: `logs/sam-ml-pandas.log`
3. **ML Scikit-Learn Agent**: `logs/sam-ml-scikit-learn.log`
4. **RAG Agent**: `logs/sam-rag.log`
5. **RAG Multi-Cloud Agent**: `logs/sam-rag-multi-cloud.log`
6. **RAG Google Drive Agent**: `logs/sam-rag-google-drive.log`
7. **MongoDB Agent**: `logs/sam-mongodb.log`

### ⚠️ Agents Still Using Generic Log File

The following agents still use the generic `solace_ai_connector.log` file:

1. **Geo Information Agent**: `solace_ai_connector.log`
2. **Solace Event Mesh Agent**: `solace_ai_connector.log`
3. **Mermaid Agent**: `solace_ai_connector.log`
4. **MCP Server Agent**: `solace_ai_connector.log`
5. **Bedrock Agent**: `solace_ai_connector.log`

## Configuration Structure

Each agent's configuration file (`configs/agents/[agent_name].yaml`) contains a logging section:

```yaml
log:
  stdout_log_level: INFO
  log_file_level: INFO
  log_file: logs/sam-[agent-name].log
```

## Benefits of Agent-Specific Logging

1. **Easier Debugging**: Each agent's logs are isolated, making it easier to troubleshoot issues
2. **Better Organization**: Logs are organized by agent rather than mixed together
3. **Performance**: No contention between agents writing to the same log file
4. **Maintenance**: Easier to manage log rotation and cleanup per agent

## Development vs Production

### Development Environment
- Log files may not be created if the logging system is configured differently
- Logs might be directed to stdout/stderr instead of files
- The `logs/` directory might not exist

### Production Environment
- Log files should be created in the specified `logs/` directory
- Each agent will write to its own log file
- Log rotation and management should be configured

## How to Verify Logging is Working

### 1. Check if logs directory exists
```bash
ls -la logs/
```

### 2. Check for log files after running agents
```bash
ls -la logs/sam-*.log
```

### 3. Monitor logs in real-time
```bash
# For SQL Database Agent
tail -f logs/sam-sql-database.log

# For ML Pandas Agent
tail -f logs/sam-ml-pandas.log

# For RAG Agent
tail -f logs/sam-rag.log
```

### 4. Check log content
```bash
# View recent logs
head -20 logs/sam-sql-database.log

# Search for specific patterns
grep "ERROR" logs/sam-ml-pandas.log
grep "Data loader action" logs/sam-ml-pandas.log
```

## Troubleshooting

### Issue: No log files are created
**Possible Causes:**
- Development environment with different logging configuration
- Logging system not configured to write to files
- Insufficient permissions to create log files

**Solutions:**
1. Check if the `logs/` directory exists and has write permissions
2. Verify the logging configuration in the agent's YAML file
3. Check if logs are being written to stdout/stderr instead

### Issue: All agents writing to the same log file
**Possible Causes:**
- Some agents still using the generic `solace_ai_connector.log` file
- Configuration not properly updated

**Solutions:**
1. Update the remaining agents to use agent-specific log files
2. Restart the agents after configuration changes
3. Verify the configuration is being loaded correctly

### Issue: Log files are empty
**Possible Causes:**
- Log level set too high (e.g., ERROR when only INFO is being logged)
- Agent not generating any log messages
- Logging system not properly initialized

**Solutions:**
1. Check the log level configuration (`stdout_log_level` and `log_file_level`)
2. Verify the agent is running and processing requests
3. Check if there are any startup errors

## Next Steps

1. **Update remaining agents** to use agent-specific log files
2. **Test logging** in your development environment
3. **Configure log rotation** for production environments
4. **Set up monitoring** to track log file sizes and errors

## Configuration Examples

### SQL Database Agent
```yaml
log:
  stdout_log_level: INFO
  log_file_level: INFO
  log_file: logs/sam-sql-database.log
```

### ML Pandas Agent
```yaml
log:
  stdout_log_level: INFO
  log_file_level: INFO
  log_file: logs/sam-ml-pandas.log
```

### RAG Agent
```yaml
log:
  stdout_log_level: INFO
  log_file_level: INFO
  log_file: logs/sam-rag.log
```

This configuration ensures that each agent writes its logs to a separate file, making it easier to debug and monitor individual agent behavior. 