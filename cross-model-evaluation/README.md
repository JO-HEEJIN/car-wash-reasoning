# Cross-model evaluation (unpublished)

Extends the paper-2 finding — a STAR reasoning scaffold raises the car
wash task's pass rate from 0% to 85% on Claude — to GPT-4o (OpenAI) and
Gemini 2.5 Pro (Google), to test whether the effect is Claude-specific
or generalizes across model families. See `cross_model_paper.md` for
the write-up and prior-result summary.

This has not been submitted anywhere. It's included here for
transparency, not as a published claim.

## Regenerating

```
pip install -r ../requirements.txt   # openai + google-genai, commented out by default
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
python experiment_cross_model.py     # GPT-4o + Gemini, full condition sweep
python experiment_gemini_only.py     # Gemini-only runs
```

Each run writes a timestamped folder under `results/` with `raw.jsonl`
and `summary.json` (and `report.md` for the full sweep runs). The
folders already in this repo are the actual runs behind
`cross_model_paper.md`'s numbers.
