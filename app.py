import json
import time
import os
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, fetch_tool


load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def build_agent(model_name: str) -> AgentExecutor:
    llm = ChatOpenAI(model=model_name)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools. Prefer calling search and fetching page content before summarizing.
                Every claim must be supported by quoted evidence and include a citation with URL and page title in sources.
                Wrap the output in this format and provide no other text\n{format_instructions}
                """
            ),
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, fetch_tool, save_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def render_markdown_from_output(output_text: str) -> tuple[str, dict | None]:
    """Parse the agent JSON and render a Markdown string. Returns (markdown, parsed_dict)."""
    try:
        data = json.loads(output_text)
    except Exception:
        # If it's not JSON, just show as-is
        return output_text, None

    if not isinstance(data, dict):
        return output_text, None

    topic = data.get("topic") or "Research Report"
    summary = data.get("summary") or ""
    sources = data.get("sources") or []
    tools_used = data.get("tools_used") or []

    lines: list[str] = []
    lines.append(f"# {topic}")
    if tools_used:
        lines.append("")
        lines.append(f"- Tools used: {', '.join(tools_used)}")
    lines.append("")
    lines.append("## Summary")
    lines.append(summary)
    lines.append("")
    lines.append("## Sources")
    if sources:
        for s in sources:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")
    return "\n".join(lines), data


def main():
    st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="centered")
    st.title("AI Research Assistant")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        model_name = st.text_input("Model", value=default_model)
        save_reports = st.checkbox("Save reports to file", value=False)

    if "agent" not in st.session_state or st.session_state.get("model_name") != model_name:
        st.session_state["agent"] = build_agent(model_name)
        st.session_state["model_name"] = model_name

    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of dicts: {role, content}

    # Chat history
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    prompt = st.chat_input("Ask a research question...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking… ⏳")
            start_ts = time.time()
            try:
                result = st.session_state["agent"].invoke({"query": prompt})
                output_text = result.get("output", "")
                md, parsed = render_markdown_from_output(output_text)
                duration = time.time() - start_ts

                placeholder.markdown(md)
                st.caption(f"Completed in {duration:.2f}s")

                if save_reports and parsed:
                    # Save the structured JSON string
                    _ = save_tool.run(json.dumps(parsed, ensure_ascii=False))

                st.session_state["messages"].append({"role": "assistant", "content": md})
            except Exception as e:
                placeholder.markdown(f"**Error:** {e}")


if __name__ == "__main__":
    main()


