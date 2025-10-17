# AI_Agents

A small research assistant using LangChain tools and OpenAI that can search the web and Wikipedia, then save a structured research result to a text file.

## Features
- Tool-calling agent with `langchain`
- Web search via DuckDuckGo and Wikipedia lookup
- Structured output validated by `pydantic`
- Saves results to `research_output.txt`
 - Fetches and summarizes full web pages for better citations

## Requirements
See `requirements.txt`. You'll also need API keys for OpenAI (and optionally Anthropic) available via environment variables.

### Environment variables
Create a `.env` file in the project root (not committed) with:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=optional_anthropic_key
```

## Setup
```bash
# From project root, ensure Python 3.11 is active
# If you use pyenv/pyenv-win:
#   pyenv install 3.11.9
#   pyenv local 3.11.9

# Create a 3.11 virtual environment
py -3.11 -m venv venv  # Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

### CLI usage
```bash
# Non-interactive run
python main.py --non-interactive --topic "Large Language Models in Finance"

# Choose model and verbose agent logs
python main.py --non-interactive --topic "Solar activity 2025" --model gpt-4o-mini --verbose
```

## Web UI (Streamlit)
```bash
# Install deps (once)
pip install -r requirements.txt

# Launch the chat UI
streamlit run app.py
```
The chat renders responses as Markdown and optionally saves structured reports to your output file.

### Configuration via environment
- **SAVE_PATH**: where to save research output file (default `research_output.txt`)
- **WIKI_MAX_CHARS**: number of characters to fetch from Wikipedia summaries (default `100`)
- **FETCH_MAX_CHARS**: max characters from fetched web pages (default `100`)

The app will prompt for a topic and then create/append the structured result to `research_output.txt`.

## GitHub
This repo includes a Python-focused `.gitignore`. The `venv/` folder and `.env` are ignored by default.
