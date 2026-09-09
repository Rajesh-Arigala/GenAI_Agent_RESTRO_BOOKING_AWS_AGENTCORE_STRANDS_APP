"""Flask chat UI for the restaurant agent.

The browser never talks to AWS. It posts to this Flask app, and the app forwards
the request to your HTTP API Gateway endpoint, which invokes the Lambda, which
invokes the agent on AgentCore Runtime.

    browser  ->  Flask (/api/chat)  ->  API Gateway  ->  Lambda  ->  AgentCore Runtime

Keeping API Gateway behind Flask means you do not need CORS on the API, and the
endpoint URL never reaches the browser.

Run it:

    pip install -r requirements.txt
    set API_GATEWAY_URL=https://xxxxx.execute-api.us-east-1.amazonaws.com/chat   (Windows)
    export API_GATEWAY_URL=https://xxxxx.execute-api.us-east-1.amazonaws.com/chat  (macOS/Linux)
    python app.py

Then open http://127.0.0.1:5000
"""

import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# The invoke URL of your HTTP API Gateway route, including the route path.
API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL", "")

# The agent can chain several model and tool calls, so give it room.
REQUEST_TIMEOUT_SECONDS = 120


@app.route("/")
def index():
    """Serve the chat page."""
    return render_template("index.html", configured=bool(API_GATEWAY_URL))


@app.route("/api/chat", methods=["POST"])
def chat():
    """Forward one turn to API Gateway and hand the reply back to the browser."""
    if not API_GATEWAY_URL:
        return jsonify({"error": "API_GATEWAY_URL is not set on the server."}), 500

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Please type a message."}), 400

    payload = {"prompt": prompt}
    for field in ("session_id", "customer_name", "today", "actor_id"):
        value = body.get(field)
        if value:
            payload[field] = value

    try:
        response = requests.post(
            API_GATEWAY_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
    except requests.Timeout:
        return jsonify({"error": "The agent took too long to reply. Try again."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Could not reach the API: {e}"}), 502

    try:
        data = response.json()
    except ValueError:
        return jsonify({"error": f"Unexpected reply from the API: {response.text[:400]}"}), 502

    if response.status_code >= 400:
        return jsonify({"error": data.get("error", "The agent returned an error.")}), response.status_code

    return jsonify({
        "result": data.get("result", ""),
        "session_id": data.get("session_id", payload.get("session_id")),
    })


if __name__ == "__main__":
    if not API_GATEWAY_URL:
        print("Warning: API_GATEWAY_URL is not set. The page will load but sending will fail.")
    app.run(host="127.0.0.1", port=5000, debug=True)
