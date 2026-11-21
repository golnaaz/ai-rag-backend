import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Allow importing backend modules when running script from project root or evals/
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from rag_service import build_rag_prompt, init_startup_document

load_dotenv(BACKEND_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=GOOGLE_API_KEY)


class JudgeScore(BaseModel):
    faithfulness: float = Field(
        ..., ge=0.0, le=1.0, description="1.0 if strictly supported by context, 0.0 if hallucinated"
    )
    relevance: float = Field(
        ..., ge=0.0, le=1.0, description="1.0 if directly answers the user question, 0.0 otherwise"
    )
    reasoning: str = Field(..., description="Short explanation of the scores")


def generate_answer(system_instruction: str, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
        ),
    )
    return response.text or ""


def judge_answer(question: str, context_prompt: str, answer: str, expected_answer: str) -> JudgeScore:
    judge_system = (
        "You are an objective RAG evaluation judge. "
        "Score faithfulness strictly against the provided reference context (no outside assumptions). "
        "Score relevance based on how well the answer addresses the question compared to the reference answer."
    )
    judge_input = (
        f"Question:\n{question}\n\n"
        f"Context & Prompt:\n{context_prompt}\n\n"
        f"Reference Answer:\n{expected_answer}\n\n"
        f"Model Answer:\n{answer}"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=judge_input,
        config=types.GenerateContentConfig(
            system_instruction=judge_system,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeScore,
        ),
    )
    return JudgeScore.model_validate_json(response.text or "{}")


def compute_page_recall(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return 1.0
    matched = len(set(retrieved_pages) & set(expected_pages))
    return round(matched / len(expected_pages), 3)


def run_evaluation(dataset_path: Path) -> None:
    init_startup_document()
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = []

    for item in items:
        question = item["question"]
        doc_id = item.get("doc_id")
        expected_answer = item.get("expected_answer", "")
        expected_pages = item.get("expected_pages", [])

        system_instruction, user_prompt, sources = build_rag_prompt(question=question, doc_id=doc_id)
        answer = generate_answer(system_instruction, user_prompt)
        score = judge_answer(question, user_prompt, answer, expected_answer)
        page_recall = compute_page_recall(sources, expected_pages)

        row = {
            "id": item.get("id", ""),
            "question": question,
            "retrieved_pages": sources,
            "expected_pages": expected_pages,
            "page_recall": page_recall,
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "reasoning": score.reasoning,
            "answer": answer,
        }
        results.append(row)
        print(
            f"[{row['id']}] faithfulness={row['faithfulness']:.2f} | "
            f"relevance={row['relevance']:.2f} | page_recall={row['page_recall']:.2f}"
        )

    if results:
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)
        avg_page = sum(r["page_recall"] for r in results) / len(results)

        print("\n--- Evaluation Summary ---")
        print(f"Total items: {len(results)}")
        print(f"Avg Faithfulness: {avg_faith:.3f}")
        print(f"Avg Relevance:    {avg_rel:.3f}")
        print(f"Avg Page Recall:  {avg_page:.3f}")

        output_file = Path(__file__).resolve().parent / "results.json"
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved detailed results to {output_file}")


if __name__ == "__main__":
    default_dataset = Path(__file__).resolve().parent / "eval_dataset.json"
    run_evaluation(default_dataset)