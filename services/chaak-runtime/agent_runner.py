import os, sys
from smolagents import MCPClient, OpenAIServerModel, ToolCallingAgent

def main():
    prompt = " ".join(sys.argv[1:]).strip() or "Dime la hora actual y calcula 17*23."
    model = OpenAIServerModel(model_id=os.environ.get("MODEL_ID", "qwen3.6-35b-a3b-q4"), api_base=os.environ.get("LLAMA_API_BASE", "http://llama-server:8080/v1"), api_key="local", flatten_messages_as_text=True)
    config = {"url": os.environ.get("MCP_URL", "http://mcp-server:8000/mcp"), "transport": "streamable-http"}
    with MCPClient(config, structured_output=True) as tools:
        print("Herramientas MCP:", ", ".join(t.name for t in tools))
        print(ToolCallingAgent(tools=tools, model=model, max_steps=6).run(prompt))

if __name__ == "__main__": main()
