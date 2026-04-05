import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
import json
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

SYSTEM_PROMPT = (
    "You are a real estate assistant. "
    "When the user asks for a property price, you MUST call the 'estimate_price' tool. "
    "Do not answer with your own knowledge, just answer with the estimated price."
    "Extract the following parameters: area_value (float), city (string), "
    "bedrooms (string), bathrooms (string), property_type (string)."
)

app = FastAPI()

# MCP Server URL
MCP_SERVER_URL = "http://127.0.0.1:8000/sse"

# Tools (that the LLM will call and use)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "estimate_price",
            "description": "Estimate property price based on area, bedrooms, bathrooms, property type and city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area_value": {"type": "number", "description": "Area in square meters"},
                    "bedrooms": {"type": "string", "description": "Number of bedrooms"},
                    "bathrooms": {"type": "string", "description": "Number of bathrooms"},
                    "property_type": {"type": "string", "description": "Type of property (Apartment, Villa, Townhouse, etc.)"},
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["area_value", "city"]   # Required since no obvious default value
            }
        }
    }
]

# Function to call the MCP server
async def call_mcp_tool(area_value: float, city: str, bedrooms: int = None, bathrooms: int = None, property_type: str = None) -> str:
    """
    Connects to the MCP server via SSE and calls the 'estimate_price' tool.
    """
    try:
        async with sse_client(url=MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "estimate_price",
                    arguments={
                        "area_value": area_value,
                        "city": city,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "property_type": property_type
                    }
                )
                if result.content and hasattr(result.content[0], "text"):
                    return result.content[0].text
                return "Unexpected response from MCP server."
    except Exception as e:
        return f"Error communicating with MCP server: {str(e)}"

# Functions to call the LLMs (depending on user choice)
async def call_ollama(user_message: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "tools": TOOLS,
            "tool_choice": "required",
            "stream": False
        }
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

async def call_openai(user_message: str, api_key: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "estimate_price"}},
            "stream": False
        }
        resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

async def call_anthropic(user_message: str, api_key: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        anthropic_tools = [{
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"]
        } for t in TOOLS]
        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
            "tools": anthropic_tools,
            "tool_choice": {"type": "tool", "name": "estimate_price"}
        }
        resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

# Main functions
async def chat_with_llm(user_message: str, provider: str, api_key: str = None) -> str:
    if provider == "ollama":
        data = await call_ollama(user_message)
        message = data.get("message", {})
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "I didn't understand.")
        args = tool_calls[0]["function"]["arguments"]
        prediction = await call_mcp_tool(
            area_value=args.get("area_value"),
            city=args.get("city"),
            bedrooms=args.get("bedrooms"),
            bathrooms=args.get("bathrooms"),
            property_type=args.get("property_type")
        )
        # Second call to reformulate
        async with httpx.AsyncClient(timeout=60.0) as client:
            final_payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": user_message},
                    message,
                    {"role": "tool", "content": prediction, "tool_call_id": tool_calls[0]["id"]}
                ],
                "stream": False
            }
            final_resp = await client.post("http://localhost:11434/api/chat", json=final_payload)
            final_resp.raise_for_status()
            return final_resp.json()["message"]["content"]

    elif provider == "openai":
        if not api_key:
            return "OpenAI API key missing."
        data = await call_openai(user_message, api_key)
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "I didn't understand.")
        args = tool_calls[0]["function"]["arguments"]
        prediction = await call_mcp_tool(args["surface"], args["rooms"], args["city"])
        # Reformulation
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            final_payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "user", "content": user_message},
                    message,
                    {"role": "tool", "content": prediction, "tool_call_id": tool_calls[0]["id"]}
                ]
            }
            final_resp = await client.post("https://api.openai.com/v1/chat/completions", json=final_payload, headers=headers)
            final_resp.raise_for_status()
            return final_resp.json()["choices"][0]["message"]["content"]

    elif provider == "claude":
        if not api_key:
            return "Anthropic API key missing."
        data = await call_anthropic(user_message, api_key)
        if data.get("stop_reason") == "tool_use":
            tool_use = next(block for block in data["content"] if block["type"] == "tool_use")
            args = tool_use["input"]
            prediction = await call_mcp_tool(args["surface"], args["rooms"], args["city"])
            return f"Estimation result: {prediction}"
        else:
            text_content = next(block["text"] for block in data["content"] if block["type"] == "text")
            return text_content
    else:
        return "Unsupported provider."

# Response of the LLM
@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    return HTMLResponse(content=HTML_PAGE)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real Estate Assistant</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .chat-box { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; margin-bottom: 10px; background: #f9f9f9; }
        .user-msg { text-align: right; color: blue; margin: 5px; }
        .bot-msg { text-align: left; color: green; margin: 5px; }
        .input-area { display: flex; gap: 10px; margin-top: 10px; }
        input, select { padding: 8px; }
        button { padding: 8px 15px; }
        .config-area { margin-bottom: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        #apiKeyInput { flex-grow: 1; }
    </style>
</head>
<body>
    <h1>🏠 Real Estate Assistant</h1>
    <div class="config-area">
        <label>LLM provider:</label>
        <select id="llmSelect" onchange="toggleApiKey()">
            <option value="ollama">Ollama (local)</option>
            <option value="openai">OpenAI (GPT-4o-mini)</option>
            <option value="claude">Anthropic Claude</option>
        </select>
        <input type="password" id="apiKeyInput" placeholder="API Key (for OpenAI/Claude)" style="display: none;">
    </div>
    <div class="chat-box" id="chatBox">
        <div class="bot-msg">Hello! Ask me about property prices.</div>
    </div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="e.g., Price of 80m² apartment with 3 bedrooms in Lyon">
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        function toggleApiKey() {
            const provider = document.getElementById('llmSelect').value;
            const apiKeyInput = document.getElementById('apiKeyInput');
            if (provider === 'ollama') {
                apiKeyInput.style.display = 'none';
                apiKeyInput.value = '';
            } else {
                apiKeyInput.style.display = 'block';
            }
        }

        async function sendMessage() {
            const inputEl = document.getElementById('userInput');
            const question = inputEl.value.trim();
            if (!question) return;
            const provider = document.getElementById('llmSelect').value;
            const apiKey = document.getElementById('apiKeyInput').value;
            const chatBox = document.getElementById('chatBox');
            const userDiv = document.createElement('div');
            userDiv.className = 'user-msg';
            userDiv.textContent = question;
            chatBox.appendChild(userDiv);
            inputEl.value = '';
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: question, provider: provider, api_key: apiKey })
            });
            const data = await response.json();
            const botDiv = document.createElement('div');
            botDiv.className = 'bot-msg';
            botDiv.textContent = data.reply;
            chatBox.appendChild(botDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Initialisation
        toggleApiKey();
    </script>
</body>
</html>
"""

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message")
    provider = data.get("provider", "ollama")
    api_key = data.get("api_key")
    if not user_message:
        return {"reply": "Please ask a question."}
    reply = await chat_with_llm(user_message, provider, api_key)
    return {"reply": reply}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)