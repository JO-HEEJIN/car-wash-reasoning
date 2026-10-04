"""
Cross-Model Car Wash Experiment
Extends the original experiment to GPT-4o and Gemini 1.5 Pro.

Conditions:
  A_bare    - No system prompt (baseline)      → 50 runs per model
  C_star    - Role + STAR reasoning framework   → 100 runs per model

Claude baseline (from prior experiments):
  A_bare: 0/20 (0%), C_star: 100/100 (100%)

Usage:
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="AIza..."
    python experiment_cross_model.py
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path

# ---- Config ----

TEMPERATURE = 0.7
QUESTION = "I want to wash my car. The car wash is 100 meters away. Should I walk or drive?"
CHALLENGE = "How will I get my car washed if I am walking?"

CONDITIONS = {
    "A_bare": {
        "system": None,
        "description": "No system prompt. Baseline.",
        "runs": 50,
    },
    "C_star": {
        "system": (
            "You are an expert advisor helping people make practical decisions. "
            "Always think through problems carefully and consider all relevant factors.\n\n"
            "When answering any question, use the STAR method:\n"
            "- Situation: What is the actual situation being described?\n"
            "- Task: What needs to be accomplished?\n"
            "- Action: What action achieves the task given the situation?\n"
            "- Result: What outcome does this action produce?\n\n"
            "Provide clear, actionable recommendations."
        ),
        "description": "Role + STAR reasoning framework.",
        "runs": 100,
    },
}

# ---- Scoring (same as original experiment.py) ----

PASS_PATTERNS = [
    r"\bshould\s+drive\b",
    r"\bshould\s+definitely\s+drive\b",
    r"\brecommend\s+driving\b",
    r"\bdrive\s+your\s+car\b",
    r"\btake\s+your\s+car\b",
    r"\bbring\s+your\s+car\b",
    r"\bdrive\s+to\s+the\s+car\s+wash\b",
    r"\bdrive\s+it\s+(there|over|to)\b",
    r"\bcar\s+needs\s+to\s+be\s+(at|there|present)\b",
    r"\bcar\s+must\s+be\s+present\b",
    r"\bneed\s+the\s+car\s+(at|there)\b",
    r"\bbest\s+to\s+drive\b",
    r"\bbetter\s+to\s+drive\b",
    r"\b(answer|solution)\s+is\s+to\s+drive\b",
]
FAIL_PATTERNS = [
    r"\bshould\s+walk\b",
    r"\bshould\s+definitely\s+walk\b",
    r"\brecommend\s+walking\b",
    r"\bwalk\s+to\s+the\s+car\s+wash\b",
    r"\bwalk\s+there\b",
    r"\bbest\s+to\s+walk\b",
    r"\bbetter\s+to\s+walk\b",
    r"\bgo\s+on\s+foot\b",
    r"\b(answer|solution)\s+is\s+to\s+walk\b",
]


def score_response(text: str) -> str:
    lower = re.sub(r'\*{1,2}', '', text.lower())
    pass_hits = sum(1 for p in PASS_PATTERNS if re.search(p, lower))
    fail_hits = sum(1 for p in FAIL_PATTERNS if re.search(p, lower))
    if pass_hits > 0 and fail_hits == 0:
        return "pass"
    elif fail_hits > 0 and pass_hits == 0:
        return "fail"
    elif pass_hits > 0 and fail_hits > 0:
        if pass_hits >= fail_hits * 2:
            return "pass"
        elif fail_hits >= pass_hits * 2:
            return "fail"
        return "ambiguous"
    else:
        return "ambiguous"


# ---- Model Clients ----

class OpenAIClient:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o"

    def ask(self, system, messages):
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            oai_messages.append({"role": m["role"], "content": m["content"]})

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=512,
            temperature=TEMPERATURE,
        )
        latency_ms = (time.time() - start) * 1000
        return response.choices[0].message.content, latency_ms


class GeminiClient:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.genai = genai
        self.model_name = "gemini-2.5-pro"

    def ask(self, system, messages):
        generation_config = self.genai.types.GenerationConfig(
            max_output_tokens=512,
            temperature=TEMPERATURE,
        )

        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system if system else None,
            generation_config=generation_config,
        )

        # Convert messages to Gemini format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [m["content"]]})

        start = time.time()
        response = model.generate_content(contents)
        latency_ms = (time.time() - start) * 1000
        return response.text, latency_ms


MODELS = {
    "gpt-4o": OpenAIClient,
    "gemini-1.5-pro": GeminiClient,
}

# Claude baseline results from prior experiments
CLAUDE_BASELINE = {
    "A_bare": {"total": 20, "primary_pass": 0, "primary_pass_rate": 0.0},
    "C_star": {"total": 100, "primary_pass": 100, "primary_pass_rate": 1.0},
}


# ---- Runner ----

def run_condition(client, model_name, cond_name, condition):
    results = []
    system = condition["system"]
    runs = condition["runs"]
    print(f"\n[{model_name} / {cond_name}] {condition['description']} ({runs} runs)")

    for i in range(runs):
        try:
            primary_text, primary_latency = client.ask(
                system, [{"role": "user", "content": QUESTION}]
            )
            primary_score = score_response(primary_text)

            challenge_text = challenge_score = challenge_latency = None
            if primary_score != "pass":
                history = [
                    {"role": "user", "content": QUESTION},
                    {"role": "assistant", "content": primary_text},
                    {"role": "user", "content": CHALLENGE},
                ]
                challenge_text, challenge_latency = client.ask(system, history)
                challenge_score = score_response(challenge_text)

            result = {
                "model": model_name,
                "condition": cond_name,
                "trial": i + 1,
                "primary_text": primary_text,
                "primary_score": primary_score,
                "primary_latency_ms": round(primary_latency),
                "challenge_text": challenge_text,
                "challenge_score": challenge_score,
                "challenge_latency_ms": round(challenge_latency) if challenge_latency else None,
            }
            results.append(result)

            tag = primary_score
            if challenge_score:
                tag += f" -> {challenge_score}"
            print(f"  [{i+1:02d}/{runs}] {tag}")

        except Exception as e:
            print(f"  [{i+1:02d}/{runs}] ERROR: {e}")
            results.append({
                "model": model_name,
                "condition": cond_name,
                "trial": i + 1,
                "primary_text": None,
                "primary_score": "error",
                "primary_latency_ms": None,
                "error": str(e),
            })

        time.sleep(0.3)

    return results


def summarize(results):
    valid = [r for r in results if r["primary_score"] != "error"]
    total = len(valid)
    if total == 0:
        return {"total": 0, "primary_pass": 0, "primary_pass_rate": 0, "errors": len(results)}
    passes = sum(1 for r in valid if r["primary_score"] == "pass")
    fails = sum(1 for r in valid if r["primary_score"] == "fail")
    ambiguous = sum(1 for r in valid if r["primary_score"] == "ambiguous")
    challenged = [r for r in valid if r.get("challenge_score") is not None]
    recovered = sum(1 for r in challenged if r["challenge_score"] == "pass")
    latencies = sorted(r["primary_latency_ms"] for r in valid if r["primary_latency_ms"])

    return {
        "total": total,
        "primary_pass": passes,
        "primary_fail": fails,
        "primary_ambiguous": ambiguous,
        "primary_pass_rate": round(passes / total, 3),
        "challenged": len(challenged),
        "recovered": recovered,
        "recovery_rate": round(recovered / len(challenged), 3) if challenged else None,
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
        "errors": len(results) - total,
    }


def write_report(out_dir, all_summaries):
    lines = [
        "# Cross-Model Car Wash Experiment Results\n",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Temperature: {TEMPERATURE}\n",
        "## Comparison Table\n",
        "| Model | Condition | Runs | Pass | Pass Rate | Recovery | Median Latency |",
        "|-------|-----------|-----:|-----:|----------:|---------:|----------------|",
    ]

    # Claude baseline first
    for cond_name in ["A_bare", "C_star"]:
        cb = CLAUDE_BASELINE[cond_name]
        lines.append(
            f"| claude-sonnet-4-5 | {cond_name} | {cb['total']} | "
            f"{cb['primary_pass']} | {cb['primary_pass_rate']:.1%} | — | — |"
        )

    # Other models
    for model_name in ["gpt-4o", "gemini-1.5-pro"]:
        for cond_name in ["A_bare", "C_star"]:
            key = f"{model_name}/{cond_name}"
            if key not in all_summaries:
                continue
            s = all_summaries[key]
            recovery = f"{s['recovery_rate']:.1%}" if s.get("recovery_rate") is not None else "—"
            latency = f"{s['median_latency_ms']}ms" if s.get("median_latency_ms") else "—"
            lines.append(
                f"| {model_name} | {cond_name} | {s['total']} | "
                f"{s['primary_pass']} | {s['primary_pass_rate']:.1%} | {recovery} | {latency} |"
            )

    lines.append("\n## Key Question\n")
    lines.append(
        "Is STAR reasoning universally effective across models, or is it Claude-specific?\n"
        "Does GPT-4o / Gemini show similar improvement from STAR prompting?"
    )

    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/cross_model_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cross-Model Car Wash Experiment")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Conditions: A_bare (50 runs), C_star (100 runs)")
    print(f"Models: gpt-4o, gemini-1.5-pro\n")

    all_results = []
    all_summaries = {}

    for model_name, ClientClass in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        try:
            client = ClientClass()
        except Exception as e:
            print(f"  SKIP: Failed to initialize {model_name}: {e}")
            continue

        for cond_name, condition in CONDITIONS.items():
            trials = run_condition(client, model_name, cond_name, condition)
            all_results.extend(trials)

            s = summarize(trials)
            s["model"] = model_name
            s["condition"] = cond_name
            s["description"] = condition["description"]
            all_summaries[f"{model_name}/{cond_name}"] = s

    # Save raw results
    with open(out_dir / "raw.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Write report
    write_report(out_dir, all_summaries)

    # Symlink latest
    latest = Path("results/cross_model_latest")
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(f"cross_model_{ts}", target_is_directory=True)

    # Print summary table
    print(f"\n\n{'='*75}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*75}")
    print(f"{'Model':<22} {'Condition':<12} {'Runs':>6} {'Pass':>6} {'Rate':>8}")
    print("-" * 55)

    # Claude baseline
    for cond in ["A_bare", "C_star"]:
        cb = CLAUDE_BASELINE[cond]
        print(f"{'claude-sonnet-4-5':<22} {cond:<12} {cb['total']:>6} {cb['primary_pass']:>6} {cb['primary_pass_rate']:>7.1%}")

    print("-" * 55)

    # New results
    for key, s in all_summaries.items():
        model, cond = key.split("/")
        print(f"{model:<22} {cond:<12} {s['total']:>6} {s['primary_pass']:>6} {s['primary_pass_rate']:>7.1%}")

    print(f"\nResults: {out_dir}/report.md")


if __name__ == "__main__":
    main()
