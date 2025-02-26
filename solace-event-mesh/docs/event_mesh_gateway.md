# Event Mesh Gateway

The Event Mesh Gateway is a powerful component of the Solace Agent Mesh Framework that enables bidirectional communication between the Solace Event Mesh and the Solace Agent Mesh. This gateway allows you to subscribe to events from the Event Mesh, process them as stimuli within the Solace Agent Mesh, and send responses back to the Event Mesh.

## Key Features

1. **Event Subscription**: The gateway subscribes to events from the Solace Event Mesh based on configured topic patterns.
2. **Event Processing**: Incoming events are processed and converted into stimuli for the Solace Agent Mesh.
3. **Response Handling**: Responses from the Solace Agent Mesh are sent back to the Event Mesh as new events.

## Configuration

The Event Mesh Gateway is configured using two main components: Event Mesh Input and Event Mesh Output.

### Event Mesh Input

The Event Mesh Input component is responsible for receiving events from the Solace Event Mesh and converting them into stimuli for the Solace Agent Mesh.

Key configuration elements:

1. **Event Handlers**: A list of handlers, each defining how to process specific types of events.
2. **Subscriptions**: Topic patterns to subscribe to for each event handler.
3. **Input Transformations**: Optional transformations to apply to incoming messages.
4. **Input Expression**: A template for processing the input payload.

Example configuration:

```yaml
event_handlers:
  - name: jira_event_handler
    subscriptions:
      - topic: jira/issue/create/>
        qos: 1
    input_expression: "template:Raise a standalone Jira Task in the EPT project that tracks the work to triage this newly created issue below. It must be in the EPT project and it should not link to any issues. Here is the new issue:{{text://input.payload}}"
    payload_encoding: utf-8
    payload_format: json
    output_handler_name: jira_output_handler
```

### Event Mesh Output

The Event Mesh Output component is responsible for sending responses from the Solace Agent Mesh back to the Solace Event Mesh as events.

Key configuration elements:

1. **Output Handlers**: A list of handlers defining how to send messages back to the Event Mesh.
2. **Topic**: The topic to publish messages to for each output handler.
3. **Payload Encoding and Format**: Specifies how to encode and format the outgoing messages.

Example configuration:

```yaml
output_handlers:
  - name: jira_output_handler
    topic: jira/issue/processed
    payload_encoding: utf-8
    payload_format: json
```

## How It Works

1. The Event Mesh Input component subscribes to specified topics on the Solace Event Mesh.
2. When an event is received, it is processed by the matching event handler.
3. The event is transformed into a stimulus for the Solace Agent Mesh using the configured input expression and transformations.
4. The Solace Agent Mesh processes the stimulus and generates a response.
5. The Event Mesh Output component takes the response and sends it back to the Solace Event Mesh using the specified output handler. This will used the configured topic, payload encoding, and format.

This bidirectional flow allows for seamless integration between your existing event-driven architecture and the powerful cognitive processing capabilities of the Solace Agent Mesh Framework.
