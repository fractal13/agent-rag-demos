#!/usr/bin/env python3

from smolagents import CodeAgent
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
from smolagents import LiteLLMModel

search_tool = DuckDuckGoSearchTool()
visit_webpage_tool = VisitWebpageTool()
tools = [ search_tool, visit_webpage_tool ]
additional_authorized_imports = []

model = LiteLLMModel(model_id="ollama_chat/qwen3:8b", api_base="http://127.0.0.1:11434")
agent = CodeAgent(
    tools=tools,
    model=model,
    additional_authorized_imports=additional_authorized_imports,
    max_steps=5
)

answer = agent.run("This webpage describes a series of events this summer. https://pokemongo.com/post/pokemon-go-road-trip-2025-gameplay?hl=en build a simple table summarizing the events, dates, times and locations.")
print(f"Agent returned answer: {answer}")
