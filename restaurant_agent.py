"""Restaurant booking agent for Amazon Bedrock AgentCore Runtime.

This single file is the whole agent:

  * four tools - three that read and write bookings in DynamoDB, one that
    searches the menu knowledge base
  * short-term memory, so the agent follows a multi-turn conversation
  * an entrypoint that AgentCore Runtime serves as an HTTP endpoint

Everything below runs inside a container that AgentCore starts for you.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import boto3
from boto3.dynamodb.conditions import Attr
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

# ---------------------------------------------------------------------------
# Configuration - every value is injected as an environment variable at launch
# time, so there are no hardcoded resource IDs in this file.
# ---------------------------------------------------------------------------
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
TABLE_NAME = os.environ.get("BOOKINGS_TABLE", "restaurant_bookings")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-2-lite-v1:0")

# AgentCore Runtime injects this automatically when the agent is configured
# with memory_mode="STM_ONLY". There is no memory id to copy around by hand.
MEMORY_ID = os.environ.get("BEDROCK_AGENTCORE_MEMORY_ID")

# How many previous turns of the conversation to replay into the model.
MEMORY_TURNS = 10

bookings_table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
kb_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
memory_client = MemoryClient(region_name=REGION) if MEMORY_ID else None

SYSTEM_PROMPT = """You are a restaurant agent. You help clients look up, create and cancel
table bookings, and answer questions about the restaurant's menu.

Rules:
- Use search_menu for any question about dishes, ingredients, prices or specials.
  Answer only from what that tool returns; never invent menu items.
- Before calling create_booking you need a date (YYYY-MM-DD), a time (HH:MM, 24h),
  a name for the reservation, and the number of guests. Ask for whatever is missing.
- After creating a booking, always tell the customer the booking id.
- When a customer asks about "my bookings" and you do not have a booking id, call
  list_bookings with their name. Only ask for their name if you do not know it yet -
  never ask for a booking id you have not given them.
- Be concise and friendly.
- Reply in short Markdown: bold for key values, bullet lists for several items. Do not
  use headings, and do not use a heading for a one-line answer."""


# ---------------------------------------------------------------------------
# Tools. The function signature and the docstring ARE the schema - Strands reads
# them and tells the model what each tool is for and what it accepts. There is
# no separate JSON schema to keep in sync.
# ---------------------------------------------------------------------------
@tool
def get_booking_details(booking_id: str) -> dict:
    """Retrieve the details of an existing restaurant booking.

    Args:
        booking_id: The ID of the booking to retrieve.
    """
    response = bookings_table.get_item(Key={"booking_id": booking_id})
    item = response.get("Item")
    if item is None:
        return {"message": f"No booking found with ID {booking_id}"}
    return item


@tool
def list_bookings(name: str) -> list:
    """List every booking held under a customer's name.

    Use this when a customer asks about "my bookings" and has not given a
    booking id.

    Args:
        name: The name the reservations were made under.
    """
    items = []
    scan_kwargs = {"FilterExpression": Attr("name").eq(name)}

    while True:
        response = bookings_table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    return items


@tool
def create_booking(date: str, name: str, hour: str, num_guests: int) -> dict:
    """Create a new restaurant booking.

    Args:
        date: The date of the booking in the format YYYY-MM-DD.
        name: Name to identify the reservation.
        hour: The hour of the booking in the format HH:MM.
        num_guests: The number of guests for the booking.
    """
    booking_id = str(uuid4())[:8]
    bookings_table.put_item(
        Item={
            "booking_id": booking_id,
            "date": date,
            "name": name,
            "hour": hour,
            "num_guests": int(num_guests),
        }
    )
    return {"booking_id": booking_id}


@tool
def delete_booking(booking_id: str) -> dict:
    """Delete an existing restaurant booking.

    Args:
        booking_id: The ID of the booking to delete.
    """
    bookings_table.delete_item(Key={"booking_id": booking_id})
    return {"message": f"Booking with ID {booking_id} deleted successfully"}


@tool
def search_menu(query: str) -> str:
    """Search the restaurant's menus and weekly specials.

    Use this for any question about dishes, ingredients, prices, the children's
    menu, the dinner menu or the specials of the week.

    Args:
        query: What the customer wants to know about the menu.
    """
    if not KNOWLEDGE_BASE_ID:
        return "The menu knowledge base is not configured."

    response = kb_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    passages = [r["content"]["text"] for r in response.get("retrievalResults", [])]
    if not passages:
        return "Nothing in the menu matches that question."
    return "\n\n---\n\n".join(passages)


TOOLS = [get_booking_details, list_bookings, create_booking, delete_booking, search_menu]


# ---------------------------------------------------------------------------
# AgentCore Memory. Read the last few turns of this session before answering,
# write the new turn afterwards. That is what lets the agent understand
# "which of those are vegetarian?" as a follow-up question.
# ---------------------------------------------------------------------------
def load_history(actor_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Return previous turns of this session in the format Strands expects."""
    if memory_client is None:
        return []

    turns = memory_client.get_last_k_turns(
        memory_id=MEMORY_ID,
        actor_id=actor_id,
        session_id=session_id,
        k=MEMORY_TURNS,
    )

    messages: List[Dict[str, Any]] = []
    for turn in reversed(turns):  # the API returns the newest turn first
        for message in turn:
            role = "user" if message.get("role", "").upper() == "USER" else "assistant"
            text = message.get("content", {}).get("text", "")
            if text:
                messages.append({"role": role, "content": [{"text": text}]})
    return messages


def save_turn(actor_id: str, session_id: str, user_text: str, agent_text: str) -> None:
    """Persist this turn so the next invocation can read it back."""
    if memory_client is None:
        return
    memory_client.create_event(
        memory_id=MEMORY_ID,
        actor_id=actor_id,
        session_id=session_id,
        messages=[(user_text, "USER"), (agent_text, "ASSISTANT")],
    )


# ---------------------------------------------------------------------------
# Entrypoint. AgentCore Runtime turns this into POST /invocations for you,
# with a /ping health check alongside it.
# ---------------------------------------------------------------------------
app = BedrockAgentCoreApp()


def build_system_prompt(customer_name: str, today: str) -> str:
    """Fold per-request context into the system prompt."""
    extras = [f"Today's date is {today}."]
    if customer_name:
        extras.append(f"The customer you are talking to is called {customer_name}.")
        extras.append("Use that name for the reservation unless they give another one.")
    return SYSTEM_PROMPT + "\n\n" + " ".join(extras)


@app.entrypoint
def invoke(payload, context):
    """Handle one turn of the conversation.

    Payload fields:
        prompt        (required) what the customer said
        customer_name (optional) name to use for reservations
        today         (optional) the date the agent should treat as today
        actor_id      (optional) who is talking - memory is scoped per actor
    """
    user_text = payload.get("prompt")
    if not isinstance(user_text, str) or not user_text.strip():
        return {"error": "'prompt' must be a non-empty string"}

    customer_name = payload.get("customer_name", "")
    today = payload.get("today") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    actor_id = payload.get("actor_id", "default-customer")
    session_id = getattr(context, "session_id", None) or "local-session"

    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
        system_prompt=build_system_prompt(customer_name, today),
        tools=TOOLS,
        messages=load_history(actor_id, session_id),
    )

    result = agent(user_text)
    try:
        agent_text = result.message["content"][0]["text"]
    except (AttributeError, KeyError, IndexError, TypeError):
        agent_text = str(result)

    save_turn(actor_id, session_id, user_text, agent_text)
    return {"result": agent_text, "session_id": session_id}


if __name__ == "__main__":
    app.run()
