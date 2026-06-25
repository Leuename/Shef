# Streaming LLM Responses Reference

## Goal

Implement a chatbot that feels responsive by streaming tokens from the
LLM instead of waiting for the complete response.

------------------------------------------------------------------------

# High-Level Architecture

``` text
User
    │
    ▼
Frontend (React/Vue/HTML/PySide6)
    │
    │ HTTP/WebSocket/SSE
    ▼
FastAPI Backend
    │
    ▼
LLM Provider
(OpenAI / NVIDIA NIM / OpenRouter / DeepSeek / Anthropic)
    │
    ▼
Token Stream
    │
    ▼
Backend yields tokens immediately
    │
    ▼
Frontend appends text as it arrives
```

------------------------------------------------------------------------

# Why Streaming Matters

Without streaming:

1.  User sends message.
2.  Backend waits for entire response.
3.  User sees nothing.
4.  Entire answer appears.

With streaming:

1.  User sends message.
2.  Model generates first token.
3.  Backend forwards token immediately.
4.  UI updates continuously.

Streaming improves perceived latency even when total generation time is
unchanged.

------------------------------------------------------------------------

# Backend Requirements

-   Use an LLM API that supports `stream=True`.
-   Never buffer the entire response.
-   Yield tokens immediately.
-   Keep a single initialized API client.
-   Use asynchronous endpoints when practical.

------------------------------------------------------------------------

# Example (OpenAI-Compatible)

``` python
from openai import OpenAI

client = OpenAI(
    base_url="YOUR_BASE_URL",
    api_key="YOUR_API_KEY"
)

stream = client.chat.completions.create(
    model="YOUR_MODEL",
    messages=[
        {"role":"user","content":"Hello"}
    ],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

------------------------------------------------------------------------

# FastAPI Example

``` python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    base_url="YOUR_BASE_URL",
    api_key="YOUR_API_KEY"
)

@app.post("/chat")
async def chat(prompt: str):

    def generate():

        stream = client.chat.completions.create(
            model="YOUR_MODEL",
            messages=[
                {"role":"user","content":prompt}
            ],
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
```

------------------------------------------------------------------------

# Frontend Responsibilities

-   Send request immediately.
-   Create an empty assistant message.
-   Append incoming chunks.
-   Auto-scroll during streaming.
-   Stop streaming cleanly if cancelled.

Pseudo-code:

``` javascript
assistantMessage = "";

for await (const chunk of stream) {
    assistantMessage += chunk;
    render();
}
```

------------------------------------------------------------------------

# Recommended Transport

## HTTP Streaming

Pros

-   Simple
-   Supported by FastAPI
-   Good default

## Server-Sent Events (SSE)

Pros

-   Excellent for chat
-   Easy to implement

## WebSockets

Pros

-   Bidirectional
-   Best for voice/chat collaboration

Use WebSockets only if true two-way communication is required.

------------------------------------------------------------------------

# Performance Recommendations

## Keep the client alive

Good

``` python
client = OpenAI(...)
```

Bad

``` python
def chat():
    client = OpenAI(...)
```

------------------------------------------------------------------------

## Async

Prefer

``` python
async def chat():
```

instead of blocking endpoints.

------------------------------------------------------------------------

## Smaller Models

Smaller models generally reduce time-to-first-token.

------------------------------------------------------------------------

## Limit Context

Instead of

    Entire PDF

retrieve only the most relevant chunks.

Example:

    Top 3–5 chunks

------------------------------------------------------------------------

## Prompt Caching

Cache repeated requests.

Example key:

    SHA256(system_prompt + conversation)

------------------------------------------------------------------------

## Retrieval Cache

Cache vector-search results.

    Question
    ↓

    Embedding
    ↓

    Top chunks

    ↓

    Reuse

------------------------------------------------------------------------

## Connection Pooling

Reuse HTTP sessions.

------------------------------------------------------------------------

## Streaming UI

Display:

-   typing cursor
-   partial markdown
-   progressive code blocks

Do not wait for completion.

------------------------------------------------------------------------

# Cancellation

Allow the user to stop generation.

When cancelled:

-   terminate stream
-   close connection
-   release resources

------------------------------------------------------------------------

# Error Handling

Handle:

-   timeout
-   provider disconnect
-   invalid API key
-   rate limit
-   malformed chunks

Show graceful UI messages instead of crashing.

------------------------------------------------------------------------

# Logging

Log:

-   request id
-   latency
-   first token latency
-   completion latency
-   token count
-   provider
-   model
-   errors

Avoid logging sensitive user content unless explicitly required.

------------------------------------------------------------------------

# Metrics

Track:

-   Time to first token (TTFT)
-   Total latency
-   Tokens/sec
-   Prompt tokens
-   Completion tokens
-   Error rate
-   Cancellation rate

------------------------------------------------------------------------

# Recommended Production Stack

-   Python 3.12+
-   FastAPI
-   Uvicorn
-   OpenAI-compatible SDK
-   LangChain (optional)
-   LangGraph (optional)
-   Redis (cache)
-   PostgreSQL
-   Chroma/Qdrant/Pinecone (RAG)
-   Nginx
-   Docker

------------------------------------------------------------------------

# Integration with LangChain

Streaming:

``` python
llm.stream(...)
```

or

``` python
agent.stream(...)
```

Forward each chunk directly to the frontend.

------------------------------------------------------------------------

# Integration with LangGraph

Each node should stream events as work progresses.

Avoid waiting until the graph finishes.

------------------------------------------------------------------------

# Common Mistakes

-   Waiting for the full response before sending.
-   Creating a new API client for every request.
-   Loading an entire PDF into context.
-   Blocking the UI.
-   Ignoring cancellation.
-   Ignoring rate limits.
-   Logging secrets.

------------------------------------------------------------------------

# Checklist

-   Streaming enabled
-   FastAPI StreamingResponse or SSE
-   Reused API client
-   Async endpoints
-   Incremental UI updates
-   Prompt cache
-   Retrieval cache
-   Logging
-   Metrics
-   Cancellation
-   Graceful error handling
-   Short prompts
-   Limited RAG context
-   Production-ready deployment

This document is intended as an implementation reference for AI coding
agents such as Codex.
