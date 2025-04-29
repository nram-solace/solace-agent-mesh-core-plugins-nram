from solace_ai_connector.common.log import log

from botocore.exceptions import ClientError
import boto3


class BedrockAgentRuntime:
    """Encapsulates Amazon Bedrock Agents Runtime actions."""

    def __init__(self, boto3_config, endpoint_url=None):
        session = boto3.Session(**boto3_config)
        self.agents_runtime_client = session.client("bedrock-agent-runtime", endpoint_url=endpoint_url)

    def invoke_agent(self, agent_id, agent_alias_id, session_id, prompt, session_state=None):
        """
        Sends a prompt for the agent to process and respond to.

        :param agent_id: The unique identifier of the agent to use.
        :param agent_alias_id: The alias of the agent to use.
        :param session_id: The unique identifier of the session. Use the same value across requests
                           to continue the same conversation.
        :param prompt: The prompt that you want model to complete.
        :param session_state: The state of the session.
        :return: Inference response from the model.
        """

        try:
            response = self.agents_runtime_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
                sessionState=session_state,
            )

            completion = ""

            for event in response.get("completion"):
                chunk = event["chunk"]
                completion = completion + chunk["bytes"].decode()

        except ClientError as e:
            log.error("Couldn't invoke agent. %s", {e})
            raise e

        return completion

    def invoke_flow(self, flow_id, flow_alias_id, input_data):
        """
        Invoke an Amazon Bedrock flow and handle the response stream.

        Args:
            param flow_id: The ID of the flow to invoke.
            param flow_alias_id: The alias ID of the flow.
            param input_data: Input data for the flow.

        Return: Response from the flow.
        """
        try:
            response = self.agents_runtime_client.invoke_flow(
                flowIdentifier=flow_id,
                flowAliasIdentifier=flow_alias_id,
                inputs=input_data,
            )

            result = ""

            # Get the streaming response
            for event in response["responseStream"]:
                result = result + str(event) + "\n"

        except ClientError as e:
            log.error("Couldn't invoke flow %s.", {e})
            raise e

        return result

