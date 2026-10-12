# LangGraph loop demo for Sulcus

A small LangGraph agent that runs a slow research loop (plan, search, reason, repeat) so you can try Sulcus in under two minutes. It needs no API key, makes no network calls, and costs nothing.

The model is a local stand-in. It reports made-up token usage (1,000 tokens per call) the way a real provider would, so Sulcus has something to count.

## Try it in Sulcus

1. Sign in at app.sulcus.dev and add a hosted run.
2. Repository: `https://github.com/sulcus-dev/langgraph-loop-demo`. Entrypoint: `agent.py`. Framework: LangGraph.
3. Start the run. It takes about a minute and a half (30 rounds) and you can watch each step live.
4. Press **Stop run** while it is going to see it end.
5. Start it again with a per-run token limit of 4,500. The run should fail after about five model calls with a token budget error.

The token limit is not a hard cap. The call that crosses the limit is allowed to finish, and the overshoot is recorded.

## Settings

Set these as environment variables if your run configuration allows it.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEMO_STEPS` | 30 | Number of research rounds |
| `DEMO_STEP_SECONDS` | 2.0 | Pause per round, in seconds |

## Run it on your own machine

```
pip install -r requirements.txt
python agent.py
```

Tested locally with langgraph 1.2.14 and langchain-core 1.6.9. It has also run to completion as a hosted Sulcus run: 30 model calls, 30,000 tokens, about 1 minute 32 seconds.
