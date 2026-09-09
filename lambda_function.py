"""AWS Lambda function that invokes the restaurant agent on AgentCore Runtime.

Sits behind an HTTP API Gateway route. This is the deployed equivalent of the
`ask()` helper in the notebook: it takes a question, forwards it to the agent
runtime with a session id, and returns the agent's reply as JSON.

All four capabilities of the agent - search the menu, create a booking, get a
booking, delete a booking - are reached through this one function, because the
agent decides which tool to call based on what the customer says.

Environment variables
---------------------
AGENT_RUNTIME_ARN   (required)  the Agent Runtime ARN printed at the end of the notebook
AGENT_QUALIFIER     (optional)  endpoint name, defaults to "DEFAULT"

IAM
---
The Lambda execution role needs:

    {
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": [
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/AGENT_ID",
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/AGENT_ID/runtime-endpoint/*"
      ]
    }

Timeout
-------
Set the Lambda timeout to at least 60 seconds. The agent may make several model
and tool calls per turn.

Request body (JSON)
-------------------
{
  "prompt":        "I want a table for 4 on 2026-05-05 at 21:00",   (required)
  "session_id":    "...",   (optional) reuse to continue a conversation
  "customer_name": "John",  (optional) name to use for the reservation
  "today":         "2026-05-01",       (optional)
  "actor_id":      "customer-123"      (optional)
}

Response body (JSON)
--------------------
{ "result": "...the agent's reply...", "session_id": "..." }
"""

import json
import os
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AGENT_QUALIFIER = os.environ.get("AGENT_QUALIFIER", "DEFAULT")

# AgentCore rejects session ids shorter than this.
MIN_SESSION_ID_LENGTH = 33

# Give the agent room to run its tool calls before boto3 gives up.
agentcore = boto3.client(
    "bedrock-agentcore",
    config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 1}),
)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}


def respond(status_code, body):
    """Build an API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def parse_body(event):
    """Read the JSON body out of an API Gateway proxy event.

    Handles the HTTP API (v2) shape, the REST API (v1) shape, and a direct
    invocation where the event *is* the payload (handy for console testing).
    """
    body = event.get("body")
    if body is None:
        return event if isinstance(event, dict) else {}

    if event.get("isBase64Encoded"):
        import base64

        body = base64.b64decode(body).decode("utf-8")

    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def new_session_id():
    """Session ids must be at least 33 characters."""
    return f"{uuid.uuid4()}-{uuid.uuid4().hex[:8]}"


def normalise_session_id(session_id):
    """Keep a caller-supplied session id if it is long enough, otherwise pad it."""
    if not session_id:
        return new_session_id()
    session_id = str(session_id)
    if len(session_id) < MIN_SESSION_ID_LENGTH:
        return f"{session_id}-{uuid.uuid4().hex}"[:64]
    return session_id


def read_agent_response(response):
    """Turn the runtime's response into the text the agent produced.

    A plain JSON response arrives as a streaming body; a streamed response
    arrives as server-sent events. Handle both so the caller sees plain text
    either way.
    """
    content_type = response.get("contentType", "")
    stream = response.get("response")

    if "text/event-stream" in content_type:
        chunks = []
        for line in stream.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data: "):
                chunks.append(decoded[6:])
        raw = "".join(chunks)
    else:
        raw = stream.read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")

    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"result": raw}

    # The agent returns {"result": ..., "session_id": ...}; a double-encoded
    # string is possible if the runtime wrapped it, so unwrap that too.
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {"result": payload}

    return payload if isinstance(payload, dict) else {"result": str(payload)}


def lambda_handler(event, context):
    # CORS preflight, in case the browser ever calls this route directly.
    method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod")
    if method == "OPTIONS":
        return respond(200, {"ok": True})

    if not AGENT_RUNTIME_ARN:
        return respond(500, {"error": "AGENT_RUNTIME_ARN environment variable is not set"})

    body = parse_body(event)
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return respond(400, {"error": "'prompt' is required and must be a non-empty string"})

    session_id = normalise_session_id(body.get("session_id"))

    # Only forward the optional fields the caller actually supplied, so the
    # agent falls back to its own defaults for the rest.
    payload = {"prompt": prompt}
    for field in ("customer_name", "today", "actor_id"):
        value = body.get(field)
        if value:
            payload[field] = value

    try:
        response = agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            qualifier=AGENT_QUALIFIER,
            runtimeSessionId=session_id,
            contentType="application/json",
            payload=json.dumps(payload),
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        message = e.response.get("Error", {}).get("Message", str(e))
        print(f"invoke_agent_runtime failed: {code}: {message}")
        status = 403 if code == "AccessDeniedException" else 502
        return respond(status, {"error": f"{code}: {message}", "session_id": session_id})
    except Exception as e:  # noqa: BLE001 - surface anything else as a 500
        print(f"Unexpected error: {e}")
        return respond(500, {"error": str(e), "session_id": session_id})

    agent_payload = read_agent_response(response)

    if "error" in agent_payload and "result" not in agent_payload:
        return respond(400, {"error": agent_payload["error"], "session_id": session_id})

    return respond(200, {
        "result": agent_payload.get("result", ""),
        "session_id": session_id,
    })
