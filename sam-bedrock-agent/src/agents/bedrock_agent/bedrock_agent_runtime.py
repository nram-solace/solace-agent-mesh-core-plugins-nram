from solace_ai_connector.common.log import log

from botocore.exceptions import ClientError
import boto3


class BedrockAgentRuntime:
    """Encapsulates Amazon Bedrock Agents Runtime actions."""

    def __init__(self, boto3_config, endpoint_url=None):
        session = boto3.Session(**boto3_config)
        self.agents_runtime_client = session.client("bedrock-agent-runtime", endpoint_url=endpoint_url)

    def invoke_agent(self, agent_id, agent_alias_id, session_id, prompt):
        """
        Sends a prompt for the agent to process and respond to.

        :param agent_id: The unique identifier of the agent to use.
        :param agent_alias_id: The alias of the agent to use.
        :param session_id: The unique identifier of the session. Use the same value across requests
                           to continue the same conversation.
        :param prompt: The prompt that you want model to complete.
        :return: Inference response from the model.
        """

        try:
            response = self.agents_runtime_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
            )

            completion = ""

            for event in response.get("completion"):
                chunk = event["chunk"]
                completion = completion + chunk["bytes"].decode()

        except ClientError as e:
            log.error("Couldn't invoke agent. %s", {e})
            raise e

        return completion

    def invoke_flow(self, flow_id, flow_alias_id, input_data, execution_id):
        """
        Invoke an Amazon Bedrock flow and handle the response stream.

        Args:
            param flow_id: The ID of the flow to invoke.
            param flow_alias_id: The alias ID of the flow.
            param input_data: Input data for the flow.
            param execution_id: Execution ID for continuing a flow. Use the value None on first run.

        Return: Response from the flow.
        """
        try:
      
            request_params = None

            if execution_id is None:
                # Don't pass execution ID for first run.
                request_params = {
                    "flowIdentifier": flow_id,
                    "flowAliasIdentifier": flow_alias_id,
                    "inputs": input_data,
                    "enableTrace": True
                }
            else:
                request_params = {
                    "flowIdentifier": flow_id,
                    "flowAliasIdentifier": flow_alias_id,
                    "executionId": execution_id,
                    "inputs": input_data,
                    "enableTrace": True
                }

            response = self.agents_runtime_client.invoke_flow(**request_params)

            if "executionId" not in request_params:
                execution_id = response['executionId']

            result = ""

            # Get the streaming response
            for event in response['responseStream']:
                result = result + str(event) + '\n'

        except ClientError as e:
            log.error("Couldn't invoke flow %s.", {e})
            raise e

        return result

