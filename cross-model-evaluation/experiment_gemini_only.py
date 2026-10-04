"""
Gemini 2.5 Pro experiment using new google-genai SDK.
GPT-4o results already collected separately.
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from google import genai

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

PASS_PATTERNS = [
    r"\bshould\s+drive\b", r"\bshould\s+definitely\s+drive\b",
    r"\brecommend\s+driving\b", r"\bdrive\s+your\s+car\b",
    r"\btake\s+your\s+car\b", r"\bbring\s+your\s+car\b",
    r"\bdrive\s+to\s+the\s+car\s+wash\b", r"\bdrive\s+it\s+(there|over|to)\b",
    r"\bcar\s+needs\s+to\s+be\s+(at|there|present)\b",
    r"\bcar\s+must\s+be\s+present\b", r"\bneed\s+the\s+car\s+(at|there)\b",
    r"\bbest\s+to\s+drive\b", r"\bbetter\s+to\s+drive\b",
    r"\b(answer|solution)\s+is\s+to\s+drive\b",
]
FAIL_PATTERNS = [
    r"\bshould\s+walk\b", r"\bshould\s+definitely\s+walk\b",
    r"\brecommend\s+walking\b", r"\bwalk\s+to\s+the\s+car\s+wash\b",
    r"\bwalk\s+there\b", r"\bbest\s+to\s+walk\b", r"\bbetter\s+to\s+walk\b",
    r"\bgo\s+on\s+foot\b", r"\b(answer|solution)\s+is\s+to\s+walk\b",
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


def ask_gemini(client, system, messages):
    config = {"max_output_tokens": 8192, "temperature": TEMPERATURE}
    if system:
        config["system_instruction"] = system

    # Build contents for multi-turn
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    start = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents,
        config=config,
    )
    latency_ms = (time.time() - start) * 1000
    return response.text, latency_ms


def run_condition(client, cond_name, condition):
    results = []
    system = condition["system"]
    runs = condition["runs"]
    print(f"\n[gemini-2.5-pro / {cond_name}] {condition['description']} ({runs} runs)", flush=True)

    for i in range(runs):
        try:
            primary_text, primary_latency = ask_gemini(
                client, system, [{"role": "user", "content": QUESTION}]
            )
            primary_score = score_response(primary_text)

            challenge_text = challenge_score = challenge_latency = None
            if primary_score != "pass":
                history = [
                    {"role": "user", "content": QUESTION},
                    {"role": "assistant", "content": primary_text},
                    {"role": "user", "content": CHALLENGE},
                ]
                challenge_text, challenge_latency = ask_gemini(client, system, history)
                challenge_score = score_response(challenge_text)

            result = {
                "model": "gemini-2.5-pro",
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
            print(f"  [{i+1:02d}/{runs}] {tag}", flush=True)

        except Exception as e:
            print(f"  [{i+1:02d}/{runs}] ERROR: {e}", flush=True)
            results.append({
                "model": "gemini-2.5-pro",
                "condition": cond_name,
                "trial": i + 1,
                "primary_text": None,
                "primary_score": "error",
                "primary_latency_ms": None,
                "error": str(e),
            })

        time.sleep(0.5)

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


def main():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/gemini_25pro_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Gemini 2.5 Pro Car Wash Experiment", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)

    all_results = []
    summaries = {}

    for cond_name, condition in CONDITIONS.items():
        trials = run_condition(client, cond_name, condition)
        all_results.extend(trials)
        s = summarize(trials)
        s["description"] = condition["description"]
        summaries[cond_name] = s

    with open(out_dir / "raw.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'='*55}", flush=True)
    print(f"{'Condition':<12} {'Runs':>6} {'Pass':>6} {'Rate':>8}", flush=True)
    print("-" * 35, flush=True)
    for cond, s in summaries.items():
        print(f"{cond:<12} {s['total']:>6} {s['primary_pass']:>6} {s['primary_pass_rate']:>7.1%}", flush=True)
    print(f"\nResults: {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
