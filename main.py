import sys
import argparse
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, fetch_tool


if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    raise RuntimeError(
        f"This project requires Python 3.11.x. Detected {sys.version.split()[0]}. "
        "Please use Python 3.11."
    )

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research assistant")
    parser.add_argument("--topic", type=str, help="Topic to research (non-interactive)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model identifier")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logs")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; require --topic")
    return parser.parse_args()


args = parse_args()

llm = ChatOpenAI(model=args.model)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (       "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. Prefer calling search and fetching page content before summarizing.
            Every claim must be supported by quoted evidence and include a citation with URL and page title in sources.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool,wiki_tool,fetch_tool,save_tool]
agent = create_tool_calling_agent(
    llm = llm, 
    prompt=prompt,
     tools=tools
    )

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=bool(args.verbose))

if args.non_interactive:
    if not args.topic:
        raise SystemExit("--non-interactive requires --topic")
    query = args.topic
else:
    query = args.topic or input("What we will Research today? ")
start_ts = time.time()
raw_response = agent_executor.invoke({"query": query})
duration_s = time.time() - start_ts

try:
    structured_response = parser.parse(raw_response["output"])
except Exception:
    print("Raw agent output:", raw_response, file=sys.stderr)
    raise

log_record = {
    "event": "research_completed",
    "topic": structured_response.topic,
    "duration_s": round(duration_s, 3),
    "tools_used": structured_response.tools_used,
}
print(json.dumps(log_record, ensure_ascii=False))
print(structured_response)