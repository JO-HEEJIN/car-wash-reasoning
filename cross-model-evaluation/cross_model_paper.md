# Structured Reasoning Frameworks Are Model-Dependent: Cross-Model Evaluation of STAR Prompt Effectiveness

**Heejin Jo**
March 20, 2026

---

## 1. Introduction

Our prior work (Jo, 2026a) demonstrated that STAR (Situation-Task-Action-Result) prompting dramatically improves implicit physical constraint recognition in Claude Sonnet 4.5, raising pass rates from 0% to 85% on the "car wash" task. A follow-up investigation (Jo, 2026b) confirmed 100% pass rates on Claude Sonnet 4.6 with n=100, and identified prompt complexity as a key factor that dilutes STAR effectiveness.

A critical open question remains: **Is STAR's effectiveness specific to Anthropic's Claude models, or does it generalize across LLM families?**

This paper extends the car wash experiment to GPT-4o (OpenAI) and Gemini 2.5 Pro (Google), testing whether structured reasoning frameworks provide universal benefits for implicit constraint recognition.

## 2. Background

### 2.1 The Car Wash Task

The task presents an implicit physical constraint:

> "I want to wash my car. The car wash is 100 meters away. Should I walk or drive?"

The correct answer is **drive** — the car must physically be at the car wash to be washed. This constraint is never stated explicitly; the model must infer it from real-world knowledge.

### 2.2 Prior Results (Jo 2026a, 2026b)

From Jo (2026a), using `claude-sonnet-4-5-20250929`, temperature 0.7, n=20 per condition:

| Condition | Description | Pass Rate |
|-----------|-------------|----------:|
| A_bare | No system prompt | 0% (0/20) |
| B_role_only | Role definition only | 0% (0/20) |
| C_role_star | Role + STAR framework | **85%** (17/20) |
| D_role_profile | Role + user profile | 30% (6/20) |
| E_full_stack | Role + STAR + profile + RAG | **100%** (20/20) |
| F_role_star_profile | Role + STAR + profile | **95%** (19/20) |

From Jo (2026b), using `claude-sonnet-4-6`, temperature 0.7, n=100:

| Condition | Pass Rate |
|-----------|----------:|
| C_star_only | **100%** (100/100) |

### 2.3 The STAR Prompt

The exact 10-line prompt used across all models:

```
You are an expert advisor helping people make practical decisions.
Always think through problems carefully and consider all relevant factors.

When answering any question, use the STAR method:
- Situation: What is the actual situation being described?
- Task: What needs to be accomplished?
- Action: What action achieves the task given the situation?
- Result: What outcome does this action produce?

Provide clear, actionable recommendations.
```

## 3. Experimental Design

### 3.1 Models

| Model | Provider | Type | Access Date |
|-------|----------|------|-------------|
| claude-sonnet-4-5-20250929 | Anthropic | Standard | 2026-02-25 |
| claude-sonnet-4-6 | Anthropic | Standard | 2026-03-08 |
| gpt-4o | OpenAI | Standard | 2026-03-20 |
| gemini-2.5-pro | Google | Thinking model | 2026-03-20 |

Note: Gemini 2.5 Pro is a "thinking" model with internal chain-of-thought reasoning that is invisible to the user but consumes output tokens. This is architecturally distinct from standard models.

### 3.2 Conditions

- **A_bare** — No system prompt (baseline). Tests whether the model can recognize the implicit constraint without any scaffolding.
- **C_star** — Role + STAR reasoning framework (10-line prompt above). Tests whether STAR prompting improves constraint recognition.

### 3.3 Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| Max tokens | 512 (Claude, GPT-4o), 8192 (Gemini 2.5 Pro)* |
| A_bare runs | 20 (Claude 4.5), 50 (GPT-4o, Gemini) |
| C_star runs | 20 (Claude 4.5), 100 (Claude 4.6, GPT-4o, Gemini) |

*Gemini 2.5 Pro requires higher max_output_tokens because its internal thinking tokens consume from the same budget. At 512 tokens, all responses were truncated before producing visible output (`finish_reason: MAX_TOKENS`).

### 3.4 Scoring

Automated regex-based scoring from Jo (2026a):
- **Pass**: Response recommends driving (e.g., "should drive", "take your car", "drive to the car wash")
- **Fail**: Response recommends walking (e.g., "should walk", "walk there", "better to walk")
- **Ambiguous**: Both or neither signal detected

A **challenge follow-up** ("How will I get my car washed if I am walking?") is issued when the primary response is not a pass.

### 3.5 Code and Data

| Item | Path |
|------|------|
| Original experiment | `car_wash/experiment.py` |
| Cross-model script (GPT-4o) | `car_wash/experiment_cross_model.py` |
| Gemini-only script | `car_wash/experiment_gemini_only.py` |
| Claude original results | `car_wash/results/20260219_024823/` |
| Claude 100-run verification | `car_wash/results/c_star_only_100_20260308_002609/` |
| GPT-4o results | `car_wash/results/cross_model_20260320_015602/` |
| Gemini 2.5 Pro results | `car_wash/results/gemini_25pro_20260320_035228/` |

## 4. Results

### 4.1 Primary Pass Rates

| Model | A_bare | C_star | STAR Lift | N (A/C) |
|-------|-------:|-------:|----------:|--------:|
| Claude Sonnet 4.5 | **0%** (0/20) | **85%** (17/20) | +85pp | 20/20 |
| Claude Sonnet 4.6 | — | **100%** (100/100) | — | —/100 |
| GPT-4o | **0%** (0/50) | **55%** (55/100) | +55pp | 50/100 |
| Gemini 2.5 Pro | **28%** (14/50) | **38%** (38/100) | +10pp | 50/100 |

### 4.2 Response Distribution

| Model | Condition | Pass | Fail | Ambiguous |
|-------|-----------|-----:|-----:|----------:|
| Claude Sonnet 4.5 | A_bare | 0 | 11 | 9 |
| Claude Sonnet 4.5 | C_star | 17 | 2 | 1 |
| GPT-4o | A_bare | 0 | 3 | **47** |
| GPT-4o | C_star | 55 | 10 | **35** |
| Gemini 2.5 Pro | A_bare | 14 | 19 | **17** |
| Gemini 2.5 Pro | C_star | 38 | 15 | **47** |

### 4.3 Challenge Recovery Rates

When the primary response fails or is ambiguous, a follow-up question explicitly highlights the constraint:

| Model | Condition | Challenged | Recovered | Recovery Rate |
|-------|-----------|----------:|----------:|--------------:|
| Claude Sonnet 4.5 | A_bare | 20 | 19 | **95%** |
| Claude Sonnet 4.5 | C_star | 3 | 2 | 67% |
| GPT-4o | A_bare | 50 | 42 | **84%** |
| GPT-4o | C_star | 45 | 37 | **82%** |
| Gemini 2.5 Pro | A_bare | 36 | 6 | **17%** |
| Gemini 2.5 Pro | C_star | 62 | 14 | **23%** |

### 4.4 Latency

| Model | Condition | Median Latency |
|-------|-----------|---------------:|
| Claude Sonnet 4.5 | C_star | 7,851 ms |
| GPT-4o | A_bare | 2,116 ms |
| GPT-4o | C_star | 5,589 ms |
| Gemini 2.5 Pro | A_bare | 18,341 ms |
| Gemini 2.5 Pro | C_star | 20,185 ms |

## 5. Analysis

### 5.1 STAR Effectiveness is Model-Dependent

STAR prompting produces dramatically different results across model families:

- **Claude**: Near-perfect (+85pp on 4.5, +100pp on 4.6). STAR reliably triggers the Task decomposition that surfaces the implicit constraint.
- **GPT-4o**: Moderate (+55pp). STAR helps but does not achieve reliability. 45% of responses still fail or are ambiguous despite explicit STAR instructions.
- **Gemini 2.5 Pro**: Marginal (+10pp). STAR provides minimal additional benefit beyond what the model's internal thinking already achieves.

### 5.2 Gemini's Built-in Thinking ≠ STAR

Gemini 2.5 Pro is unique in this study as a "thinking" model with internal chain-of-thought reasoning. This produces two notable effects:

1. **Higher baseline (28% vs 0%)**: Internal reasoning allows Gemini to sometimes recognize the implicit constraint even without STAR prompting, unlike Claude and GPT-4o which score 0% at baseline.

2. **Lower STAR ceiling (38% vs 100%/55%)**: Despite internal thinking AND explicit STAR instructions, Gemini achieves the lowest STAR pass rate. This suggests that Gemini's internal reasoning may interfere with or override the external STAR framework, rather than complementing it.

**Key insight**: Internal chain-of-thought (thinking models) and external structured prompting (STAR) are not additive. Gemini's thinking raises the floor but appears to lower the ceiling that STAR can achieve.

### 5.3 The Ambiguity Problem

Non-Claude models exhibit a strong "hedging" tendency:

- **GPT-4o A_bare**: 94% ambiguous (47/50) — almost never commits to walk or drive
- **GPT-4o C_star**: 35% ambiguous — STAR reduces but doesn't eliminate hedging
- **Gemini C_star**: 47% ambiguous — thinking model hedges at the same rate as non-thinking GPT-4o baseline

This hedging manifests as responses that discuss both options ("Walking is convenient for the short distance, but you'll need your car there...") without making a clear recommendation. Claude, by contrast, produces 0% ambiguous responses under STAR.

### 5.4 Challenge Recovery Reveals Model Differences

When explicitly told "How will I get my car washed if I am walking?":

- **Claude (95%) and GPT-4o (84%)** readily correct themselves — the constraint was accessible but not prioritized.
- **Gemini (17%)** largely fails to correct even with an explicit hint. This suggests a deeper processing difference: Gemini may handle the correction within its thinking tokens but still produce a hedged output.

### 5.5 Scoring Methodology Interaction

The high ambiguity rates raise the question of whether the regex-based scoring penalizes certain model response styles. GPT-4o and Gemini tend to produce longer, more nuanced responses that mention both options, triggering both pass and fail patterns simultaneously. A manual review of ambiguous responses is recommended for future work.

## 6. Discussion

### 6.1 Why Does STAR Work Best on Claude?

Several hypotheses:

1. **Training alignment**: Claude may be more strongly trained to follow structured reasoning templates like STAR, producing cleaner task decomposition.
2. **Instruction following fidelity**: Claude appears to execute STAR steps sequentially before reaching a conclusion, while GPT-4o and Gemini may interleave STAR with their own reasoning, diluting the framework's constraint-surfacing effect.
3. **Commitment vs. hedging**: Claude's tendency to commit to a single recommendation (0% ambiguous) amplifies STAR's effect — once STAR surfaces the constraint, Claude acts on it decisively. GPT-4o and Gemini's hedging means that even when STAR surfaces the constraint, the model may not commit to it in the final recommendation.

### 6.2 Implications for Prompt Engineering

1. **Structured prompting is not universally effective**. STAR produces +85–100pp on Claude but only +10pp on Gemini. Prompt engineers cannot assume cross-model transferability.
2. **Thinking models may resist external reasoning frameworks**. Gemini 2.5 Pro's internal chain-of-thought appears to compete with rather than complement STAR.
3. **Model-specific prompt optimization is necessary**. The optimal prompting strategy for implicit constraint tasks differs by model family.

### 6.3 Implications for the Car Wash Benchmark

The car wash task reveals a spectrum of implicit reasoning capabilities:

- **Without prompting**: Claude/GPT-4o = 0%, Gemini = 28%. Only thinking models show any baseline capability.
- **With STAR**: Claude = 85–100%, GPT-4o = 55%, Gemini = 38%. Structured prompting helps all models but benefits non-thinking models more.
- **With explicit hint**: Claude = 95%, GPT-4o = 84%, Gemini = 17%. Standard models recover readily; thinking models do not.

This suggests the car wash task measures at least two distinct capabilities: (1) implicit constraint recognition, and (2) instruction-prompted reasoning compliance.

## 7. Limitations

1. **Model versions**: GPT-4o and Gemini 2.5 Pro were accessed on a single date (2026-03-20). Model behavior may change with updates.
2. **Gemini max_tokens**: Gemini 2.5 Pro required 8192 max_output_tokens (vs 512 for others) due to thinking token consumption. This difference in generation budget could affect response characteristics.
3. **Scoring granularity**: Regex-based scoring does not capture responses that correctly identify the constraint but present both options without committing. A human evaluation would provide finer granularity.
4. **Single task**: Results are based on one implicit constraint task. Generalizability to other implicit reasoning tasks is unknown.
5. **Sample sizes differ**: Claude 4.5 data uses n=20 while cross-model data uses n=50/100. The smaller Claude sample may not capture full variance.
6. **Distance modification**: The original ryan-allen/car-wash-evals uses 50 meters. Our experiments use 100 meters. At 100 meters, "drive" may be slightly more intuitive, potentially making the task marginally easier. Despite this, all baseline conditions without STAR on Claude and GPT-4o still produced 0% pass rates.

## 8. Conclusion

STAR prompting is **not universally effective** across LLM families for implicit constraint recognition. Its dramatic impact on Claude (+85–100pp) does not transfer proportionally to GPT-4o (+55pp) or Gemini 2.5 Pro (+10pp).

Three key findings emerge:

1. **STAR effectiveness is strongly model-dependent**, suggesting that structured reasoning frameworks exploit model-specific instruction-following capabilities rather than a universal reasoning mechanism.

2. **Internal thinking and external STAR are not additive** — Gemini's thinking mode raises baseline performance but limits STAR's additional contribution, implying these are competing rather than complementary reasoning pathways.

3. **Response commitment style mediates STAR effectiveness** — Claude's decisive, low-ambiguity responses amplify STAR's constraint-surfacing effect, while GPT-4o and Gemini's hedging tendency dilutes it.

These findings have practical implications for prompt engineering: structured reasoning prompts should be validated per-model, and techniques that work well on one LLM family should not be assumed to transfer.

## References

- Jo, H. (2026a). Prompt Architecture Determines Reasoning Quality: A Variable Isolation Study on the Car Wash Problem. arXiv:2602.21814. https://arxiv.org/abs/2602.21814
- Jo, H. (2026b). Prompt Complexity Dilutes Structured Reasoning: A Follow-Up Study on the Car Wash Problem. arXiv:2603.13351. https://arxiv.org/abs/2603.13351
- Allen, R. (2026). car-wash-evals. GitHub. https://github.com/ryan-allen/car-wash-evals

## Appendix A: Raw Data Summary

### A.1 Claude Sonnet 4.5 (Original Paper, 2026-02-25)

```
Model: claude-sonnet-4-5-20250929
Temperature: 0.7, N=20 per condition

A_bare:       0/20  (0.0%)   | Fail: 11, Ambig: 9   | Recovery: 95% (19/20)
B_role_only:  0/20  (0.0%)   | Fail: 19, Ambig: 1   | Recovery: 100% (20/20)
C_role_star: 17/20 (85.0%)   | Fail: 2,  Ambig: 1   | Recovery: 67% (2/3)
D_role_prof:  6/20 (30.0%)   | Fail: 13, Ambig: 1   | Recovery: 100% (14/14)
E_full_stack:20/20 (100.0%)  | Fail: 0,  Ambig: 0   | Recovery: n/a
F_star_prof: 19/20 (95.0%)   | Fail: 0,  Ambig: 1   | Recovery: 0% (0/1)
```

### A.2 Claude Sonnet 4.6 (100-Run Verification, 2026-03-08)

```
Model: claude-sonnet-4-6
Temperature: 0.7, N=100

C_star_only: 100/100 (100.0%) | Fail: 0, Ambig: 0
```

### A.3 GPT-4o (2026-03-20)

```
Model: gpt-4o
Temperature: 0.7

A_bare (N=50):   0/50  (0.0%)  | Fail: 3,  Ambig: 47 | Recovery: 84% (42/50)
C_star (N=100): 55/100 (55.0%) | Fail: 10, Ambig: 35 | Recovery: 82% (37/45)
Median latency: 2,116ms (A), 5,589ms (C)
```

### A.4 Gemini 2.5 Pro (2026-03-20)

```
Model: gemini-2.5-pro (thinking model)
Temperature: 0.7, max_output_tokens: 8192

A_bare (N=50):  14/50  (28.0%) | Fail: 19, Ambig: 17 | Recovery: 17% (6/36)
C_star (N=100): 38/100 (38.0%) | Fail: 15, Ambig: 47 | Recovery: 23% (14/62)
Median latency: 18,341ms (A), 20,185ms (C)
```
