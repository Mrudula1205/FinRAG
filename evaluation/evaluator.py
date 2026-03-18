"""
Evaluation pipeline: LLM-as-Judge scoring on four dimensions.

Judge model: openai/gpt-oss-120b (via Groq)
  - Different architecture from the RAG LLM (llama-3.3-70b) — avoids self-evaluation bias.
  - Deterministic scoring (temperature=0).

Dimensions evaluated (1–5 scale):
  relevance    — retrieved context addresses the query
  correctness  — answer is factually supported by the context
  completeness — answer covers all important aspects
  faithfulness — answer contains only claims grounded in the context
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd
from langchain_groq import ChatGroq

from config.settings import (
    DEFAULT_TOP_K,
    EVAL_RESULTS_PATH,
    JUDGE_LLM_MODEL,
    JUDGE_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from rag.pipeline import extract_content_for_rag
from rag.retriever import RAGRetriever

# ── Default evaluation dataset ────────────────────────────────────────────────

EVAL_DATASET = [
    # Easy — straightforward lookup
    {"query": "What is the maximum fine under GDPR for serious violations?",                                                                      "difficulty": "easy"},
    {"query": "Under GDPR, how many days does a controller have to report a data breach to the supervisory authority?",                           "difficulty": "easy"},
    {"query": "What rights does a data subject have under GDPR?",                                                                                 "difficulty": "easy"},
    # Medium — application / reasoning
    {"query": "We collect email addresses for a newsletter. What lawful basis should we rely on under GDPR?",                                     "difficulty": "medium"},
    {"query": "We had a minor data breach affecting 200 users — do we need to notify authorities?",                                               "difficulty": "medium"},
    {"query": "An employee requests a copy of all personal data we hold about them. What are our obligations?",                                   "difficulty": "medium"},
    {"query": "We store customer emails on AWS US-East servers. Does this constitute an international data transfer?",                            "difficulty": "medium"},
    # Hard — multi-step / nuanced
    {"query": "We process health data of EU citizens for medical research. What additional safeguards does GDPR require for special category data?", "difficulty": "hard"},
    {"query": "Can we use legitimate interest as the legal basis for direct marketing under GDPR? What conditions must be met?",                   "difficulty": "hard"},
    {"query": "What are the requirements for a valid Data Processing Agreement (DPA) between a controller and a processor under GDPR?",           "difficulty": "hard"},
]

# ── Prompt template ───────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an expert evaluator for GDPR compliance RAG systems.
Score the following on FOUR dimensions, each from 1 to 5.

## Query
{query}

## Retrieved Context
{context}

## Generated Answer
{answer}

---
Score each dimension strictly as an integer 1-5:
- 1 = very poor  2 = poor  3 = acceptable  4 = good  5 = excellent

Return ONLY a valid JSON object with this exact structure (no extra text):
{{
  "relevance":     {{"score": <1-5>, "reason": "<one sentence>"}},
  "correctness":   {{"score": <1-5>, "reason": "<one sentence>"}},
  "completeness":  {{"score": <1-5>, "reason": "<one sentence>"}},
  "faithfulness":  {{"score": <1-5>, "reason": "<one sentence>"}}
}}

Definitions:
- relevance:    Does the retrieved context contain information needed to answer the query?
- correctness:  Is the generated answer factually supported by the retrieved context (no hallucinations)?
- completeness: Does the answer address all important aspects of the query?
- faithfulness: Does the answer contain ONLY claims grounded in the retrieved context above?
               Penalise any statement that introduces facts, article numbers, obligations, or thresholds
               NOT present in the context (hallucinated external knowledge from the model's memory).
               Score 5 = every claim traceable to context; Score 1 = major fabricated content.
"""


# ── Judge evaluator ───────────────────────────────────────────────────────────

class LLMJudgeEvaluator:
    """Score RAG outputs on relevance, correctness, completeness, and faithfulness."""

    def __init__(self, judge_llm: ChatGroq | None = None, api_key: str | None = None):
        if judge_llm is not None:
            self.judge_llm = judge_llm
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.judge_llm = ChatGroq(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                model_name=JUDGE_LLM_MODEL,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        print(f"✅ Judge LLM ready: {self.judge_llm.model_name}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_scores(self, raw: str) -> dict:
        """Extract the last top-level JSON block from the judge's output."""
        matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL))
        if not matches:
            raise ValueError(f"No JSON found in judge output:\n{raw}")
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"Malformed JSON in judge output:\n{raw}")
            return json.loads(match.group())

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, query: str, context: str, answer: str) -> dict:
        """
        Score a single (query, context, answer) triple.

        Returns a flat dict with ``*_score`` and ``*_reason`` keys plus ``mean_score``.
        """
        prompt   = _JUDGE_PROMPT.format(query=query, context=context, answer=answer)
        response = self.judge_llm.invoke(prompt)
        scores   = self._parse_scores(response.content)

        return {
            "query":               query,
            "relevance_score":     scores["relevance"]["score"],
            "relevance_reason":    scores["relevance"]["reason"],
            "correctness_score":   scores["correctness"]["score"],
            "correctness_reason":  scores["correctness"]["reason"],
            "completeness_score":  scores["completeness"]["score"],
            "completeness_reason": scores["completeness"]["reason"],
            "faithfulness_score":  scores["faithfulness"]["score"],
            "faithfulness_reason": scores["faithfulness"]["reason"],
            "mean_score": round(
                (
                    scores["relevance"]["score"]
                    + scores["correctness"]["score"]
                    + scores["completeness"]["score"]
                    + scores["faithfulness"]["score"]
                ) / 4,
                2,
            ),
        }


# ── Pipeline runner ───────────────────────────────────────────────────────────

_RAG_PROMPT_TEMPLATE = (
    "You are a GDPR compliance expert. Answer the question using ONLY the provided context.\n"
    "If the context does not contain sufficient information to answer, say "
    '"The provided documents do not address this directly."\n'
    "Cite the relevant Article and Clause numbers in your answer where available.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


def run_evaluation(
    dataset: list,
    retriever: RAGRetriever,
    rag_llm: ChatGroq,
    judge: LLMJudgeEvaluator,
    top_k: int = DEFAULT_TOP_K,
    sleep_between: float = 1.5,
    save_path=EVAL_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Evaluate the RAG pipeline against *dataset*.

    For each test case:
      1. Retrieve context once (used for both generation and judging).
      2. Generate answer with *rag_llm*.
      3. Score with *judge* on four dimensions.

    Args:
        save_path: If provided, the resulting DataFrame is saved as CSV here.

    Returns:
        DataFrame with one row per test case.
    """
    results = []
    total   = len(dataset)

    for idx, item in enumerate(dataset, 1):
        query      = item["query"]
        difficulty = item.get("difficulty", "—")
        print(f"[{idx:02d}/{total}] ({difficulty}) {query[:70]}...")

        # Retrieve
        docs    = retriever.retrieve(query, top_k=top_k, score_threshold=0.1)
        context = extract_content_for_rag(docs) if docs else ""
        top_sim = round(max((d["similarity_score"] for d in docs), default=0.0), 3)

        # Generate
        if not docs:
            rag_answer = "No relevant documents found."
        else:
            rag_answer = rag_llm.invoke(
                _RAG_PROMPT_TEMPLATE.format(context=context, query=query)
            ).content

        # Judge
        try:
            scores = judge.evaluate(query, context, rag_answer)
        except Exception as exc:
            print(f"  ⚠️  Judge failed: {exc}")
            scores = {
                "query":               query,
                "relevance_score":     None, "relevance_reason":    str(exc),
                "correctness_score":   None, "correctness_reason":  None,
                "completeness_score":  None, "completeness_reason": None,
                "faithfulness_score":  None, "faithfulness_reason": None,
                "mean_score":          None,
            }

        results.append({
            "difficulty":    difficulty,
            "top_sim_score": top_sim,
            "rag_answer":    rag_answer,
            **scores,
        })
        print(
            f"       relevance={scores['relevance_score']}  "
            f"correctness={scores['correctness_score']}  "
            f"completeness={scores['completeness_score']}  "
            f"faithfulness={scores['faithfulness_score']}  "
            f"mean={scores['mean_score']}"
        )
        time.sleep(sleep_between)

    df = pd.DataFrame(results)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 Results saved to {save_path}")

    print(f"\n✅ Evaluation complete — {len(results)} cases scored")
    return df


# ── Scorecard printer ─────────────────────────────────────────────────────────

def print_scorecard(df: pd.DataFrame) -> None:
    """Print a formatted summary scorecard to stdout."""
    score_cols = [
        "relevance_score", "correctness_score",
        "completeness_score", "faithfulness_score", "mean_score",
    ]
    display_cols = ["query", "difficulty", "top_sim_score"] + score_cols

    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("=" * 100)
    print("EVALUATION SCORECARD")
    print("=" * 100)
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("AGGREGATE SUMMARY")
    print("=" * 100)
    print(df[score_cols].agg(["mean", "min", "max"]).round(2).to_string())

    print("\n── Scores by difficulty ──")
    print(df.groupby("difficulty")[score_cols].mean().round(2).to_string())

    print("\n── Pass rate (score ≥ 4) ──")
    for col in score_cols[:-1]:
        pass_rate = (df[col] >= 4).mean() * 100
        print(f"  {col:<25}: {pass_rate:.0f}%")

    weak = df[df["mean_score"] < 3][
        ["query", "mean_score", "relevance_score", "correctness_score",
         "completeness_score", "faithfulness_score"]
    ]
    if not weak.empty:
        print(f"\n⚠️  {len(weak)} case(s) with mean score < 3:")
        print(weak.to_string(index=False))
    else:
        print("\n✅ No cases scored below 3 — pipeline looks healthy!")
