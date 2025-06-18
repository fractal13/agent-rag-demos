#!/usr/bin/env python3

from smolagents import CodeAgent
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
from smolagents import LiteLLMModel

from rag_tools import KnowledgeBaseReadTool, KnowledgeBaseWriteTool
# Instantiate the RAG tools
read_tool = KnowledgeBaseReadTool()
write_tool = KnowledgeBaseWriteTool()


search_tool = DuckDuckGoSearchTool()
visit_webpage_tool = VisitWebpageTool()
tools = [ search_tool, visit_webpage_tool, read_tool, write_tool ]
additional_authorized_imports = []

model = LiteLLMModel(model_id="ollama_chat/qwen3:8b", api_base="http://127.0.0.1:11434")
agent = CodeAgent(
    tools=tools,
    model=model,
    additional_authorized_imports=additional_authorized_imports,
    max_steps=20
)

user_prompt = """The site https://pokemongo.com/en/news contains news on officially published Pokemon GO events.
You also have a knowledge base with information about events previously discovered.
Check the site.
For any events that are not already stored in the knowledge base, read the details of the event, and store the information in the knowledge base.
Finally, give a reverse chronological list of all events and their dates. 
Be sure to pay attention to the date of the event, not the date of the event's announcement.
"""

answer = agent.run(user_prompt)
print(f"Agent returned answer: {answer}")
