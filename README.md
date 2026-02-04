<h1 align="center">Pydantic AI RLM</h1>

<p align="center">
  <b>Handle Extremely Large Contexts with Any LLM Provider</b>
</p>

<p align="center">
  <a href="https://github.com/vstorm-co/pydantic-ai-rlm">GitHub</a> •
  <a href="https://pypi.org/project/pydantic-ai-rlm/">PyPI</a> •
  <a href="https://github.com/vstorm-co/pydantic-ai-rlm#examples">Examples</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
  <a href="https://pypi.org/project/pydantic-ai-rlm/"><img src="https://img.shields.io/pypi/v/pydantic-ai-rlm.svg" alt="PyPI version"></a>
</p>

<p align="center">
  <b>Switch Providers Instantly</b>
  &nbsp;•&nbsp;
  <b>Sandboxed Code Execution</b>
  &nbsp;•&nbsp;
  <b>Sub-Model Delegation</b>
  &nbsp;•&nbsp;
  <b>Grounded Citations</b>
  &nbsp;•&nbsp;
  <b>Fully Type-Safe</b>
</p>

---

## What is RLM?

**RLM (Recursive Language Model)** is a pattern for handling contexts that exceed a model's context window, introduced by **Alex L. Zhang, Tim Kraska, and Omar Khattab** in their paper [Recursive Language Models](https://arxiv.org/abs/2512.24601). Instead of trying to fit everything into one prompt, the LLM writes Python code to programmatically explore and analyze the data.

**The key insight:** An LLM can write code to search through millions of lines in seconds, then use `llm_query()` to delegate semantic analysis of relevant chunks to a sub-model.

This library is an implementation inspired by the [original minimal implementation](https://github.com/alexzhang13/rlm-minimal).

---

## Get Started in 60 Seconds

```bash
pip install pydantic-ai-rlm
```

```python
from pydantic_ai_rlm import run_rlm_analysis

answer = await run_rlm_analysis(
    context=massive_document,  # Can be millions of characters
    query="Find the magic number hidden in the text",
    model="openai:gpt-5",
    sub_model="openai:gpt-5-mini",
)
```

**That's it.** Your agent can now:

- Write Python code to analyze massive contexts
- Use `llm_query()` to delegate semantic analysis to sub-models
- Work with any Pydantic AI compatible provider

---

## Why pydantic-ai-rlm?

### Switch Providers Instantly

Built on Pydantic AI, you can test any model with a single line change:

```python
# OpenAI
agent = create_rlm_agent(model="openai:gpt-5", sub_model="openai:gpt-5-mini")

# Anthropic
agent = create_rlm_agent(model="anthropic:claude-sonnet-4-5", sub_model="anthropic:claude-haiku-4-5")

agent = create_rlm_agent(model="anthropic:claude-sonnet-4-5", sub_model="openai:gpt-5-mini")
```

### Reusable Toolset

The RLM toolset integrates with any pydantic-ai agent:

```python
from pydantic_ai import Agent
from pydantic_ai_rlm import create_rlm_toolset, RLMDependencies

# Use the toolset in any agent
toolset = create_rlm_toolset(sub_model="openai:gpt-5-mini")
agent = Agent("openai:gpt-5", toolsets=[toolset])
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         pydantic-ai-rlm                         │
│                                                                 │
│   ┌─────────────┐         ┌─────────────────────────────────┐   │
│   │   Main LLM  │         │     Sandboxed REPL Environment  │   │
│   │   (gpt-5)   │────────>│                                 │   │
│   └─────────────┘         │   context = <your massive data> │   │
│         │                 │                                 │   │
│         │                 │   # LLM writes Python code:     │   │
│         │                 │   for line in context.split():  │   │
│         │                 │       if "magic" in line:       │   │
│         │                 │           result = llm_query(   │   │
│         │                 │               f"Analyze: {line}"│   │
│         │                 │           )                     │   │
│         │                 │                                 │   │
│         │                 └───────────────┬─────────────────┘   │
│         │                                 │                     │
│         │                                 ▼                     │
│         │                       ┌─────────────────┐             │
│         │                       │    Sub LLM      │             │
│         │                       │  (gpt-5-mini)   │             │
│         │                       └─────────────────┘             │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────┐                                               │
│   │   Answer    │                                               │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Main LLM** receives the query and writes Python code
2. **REPL Environment** executes code with access to `context` variable
3. **llm_query()** delegates semantic analysis to a cheaper/faster sub-model
4. **Main LLM** synthesizes the final answer from code execution results

---

## Examples

### Needle in Haystack

Find specific information in massive text:

```python
from pydantic_ai_rlm import run_rlm_analysis

# 1 million lines of text with a hidden number
massive_text = generate_haystack(num_lines=1_000_000)

answer = await run_rlm_analysis(
    context=massive_text,
    query="Find the magic number hidden in the text",
    model="openai:gpt-5",
    sub_model="openai:gpt-5-mini",
)
```

### JSON Data Analysis

Works with structured data too:

```python
from pydantic_ai_rlm import create_rlm_agent, RLMDependencies

agent = create_rlm_agent(model="openai:gpt-5")

deps = RLMDependencies(
    context={"users": [...], "transactions": [...], "logs": [...]},
)

result = await agent.run(
    "Find all users with suspicious transaction patterns",
    deps=deps,
)
```

### Grounded Responses with Citations

Get answers with traceable citations back to the source:

```python
from pydantic_ai_rlm import run_rlm_analysis

# Enable grounding for citation tracking
result = await run_rlm_analysis(
    context=financial_report,
    query="What were the key revenue changes?",
    model="openai:gpt-5",
    grounded=True,  # Returns GroundedResponse instead of str
)

# Response contains citation markers
print(result.info)
# "Revenue increased [1] primarily due to [2]"

# Grounding maps markers to exact quotes from the source
print(result.grounding)
# {"1": "by 45% year-over-year", "2": "expansion into Asian markets"}
```

---

## API Reference

### `create_rlm_agent()`

Create a Pydantic AI agent with RLM capabilities.

```python
agent = create_rlm_agent(
    model="openai:gpt-5",           # Main model for orchestration
    sub_model="openai:gpt-5-mini",  # Model for llm_query() (optional)
    code_timeout=60.0,               # Timeout for code execution
    custom_instructions="...",       # Additional instructions
    grounded=True,                   # Return GroundedResponse with citations
)
```

### `create_rlm_toolset()`

Create a standalone RLM toolset for composition.

```python
toolset = create_rlm_toolset(
    code_timeout=60.0,
    sub_model="openai:gpt-5-mini",
)
```

### `run_rlm_analysis()` / `run_rlm_analysis_sync()`

Convenience functions for quick analysis.

```python
# Async
answer = await run_rlm_analysis(context, query, model="openai:gpt-5")

# Sync
answer = run_rlm_analysis_sync(context, query, model="openai:gpt-5")

# With grounding (returns GroundedResponse)
result = await run_rlm_analysis(context, query, grounded=True)
print(result.info)       # Text with [N] markers
print(result.grounding)  # {"1": "exact quote", ...}
```

### `RLMDependencies`

Dependencies for RLM agents.

```python
deps = RLMDependencies(
    context="...",  # str, dict, or list
    config=RLMConfig(
        code_timeout=60.0,
        truncate_output_chars=50_000,
        sub_model="openai:gpt-5-mini",
    ),
)
```

### `configure_logging()`

Enable verbose logging to see what the agent is doing in real-time.

```python
from pydantic_ai_rlm import configure_logging, run_rlm_analysis

# Enable logging (uses rich if installed, falls back to plain text)
configure_logging(enabled=True)

# Now you'll see code executions and outputs in the terminal
answer = await run_rlm_analysis(
    context=massive_document,
    query="Find the magic number",
    model="openai:gpt-5",
)

# Disable logging when done
configure_logging(enabled=False)
```

Install with rich logging support for syntax highlighting and styled output:

```bash
pip install pydantic-ai-rlm[logging]
```

Or install rich separately:

```bash
pip install rich
```

When enabled, you'll see:

- Syntax-highlighted code being executed (with rich)
- Execution results with status indicators (SUCCESS/ERROR)
- Execution time for each code block
- Variables created during execution
- LLM sub-queries and responses (when using `llm_query()`)

**Note:** Logging works without rich installed - it will use plain text output instead of styled panels

---

## REPL Environment

The sandboxed REPL provides:

| Feature | Description |
|---------|-------------|
| `context` variable | Your data loaded and ready to use |
| `llm_query(prompt)` | Delegate to sub-model (if configured) |
| Safe built-ins | `print`, `len`, `range`, etc. |
| Common imports | `json`, `re`, `collections`, etc. |
| Persistent state | Variables persist across executions |
| Output capture | stdout/stderr returned to agent |

**Blocked for security:** `eval`, `exec`, `compile`, `open` (outside temp dir)

---

## Related Projects

- **[rlm](https://github.com/alexzhang13/rlm)** - Original RLM implementation by Alex L. Zhang, Tim Kraska, and Omar Khattab
- **[rlm-minimal](https://github.com/alexzhang13/rlm-minimal)** - Minal RLM implementation by Alex L. Zhang
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** - The foundation: Agent framework by Pydantic
- **[pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents)** - Full deep agent framework with planning, filesystem, and more

---

## Contributing

```bash
git clone https://github.com/vstorm-co/pydantic-ai-rlm.git
cd pydantic-ai-rlm
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE)

<p align="center">
  <sub>Built with Pydantic AI by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
