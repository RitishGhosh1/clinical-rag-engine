import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows platforms
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import get_settings
from app.schemas import DocumentMetadatafilter
from app.services.rag_engine import RagEngine

settings = get_settings()

def run_tests():
    print("=" * 70)
    print("CLINICAL TRIAL RAG ENGINE — VERIFICATION SUITE")
    print("=" * 70)

    # Sanity check: Ensure Groq API Key exists
    if not settings.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not found in environment or .env file.")
        return

    rag = RagEngine()

    test_cases = [
        {
            "name": "TEST 1: In-Domain Query (Auto-Extracted Drug & Phase Metadata)",
            "question": "What are the primary considerations for sample size and statistical power in Phase III trials of Pembrolizumab?",
            "filter": None,
            "expected_behavior": "Should auto-extract drug ('Pembrolizumab') and phase ('Phase III'), retrieve statistical sizing chunks (Pages 62-63), and synthesize an auditable response with inline citations.",
            "validate": lambda ans, cits, count, f, q: (
                count > 0 
                and len(cits) > 0 
                and f is not None 
                and f.trial_phase == "Phase III" 
                and f.drug_name == "Pembrolizumab"
                and "Insufficient clinical documentation" not in ans
            )
        },
        {
            "name": "TEST 2: In-Domain Query (Explicit Filter Override)",
            "question": "What are the reporting timelines for serious unexpected adverse events in IND safety reports?",
            "filter": DocumentMetadatafilter(drug_name="Pembrolizumab", trial_phase="Phase III"),
            "expected_behavior": "Should bypass auto-extraction, apply explicit filter directly, retrieve safety reporting chunks (Pages 92-93), and cite the 7-day and 60-day reporting deadlines.",
            "validate": lambda ans, cits, count, f, q: (
                count > 0 
                and len(cits) > 0 
                and f is not None 
                and f.trial_phase == "Phase III" 
                and "Insufficient clinical documentation" not in ans
            )
        },
        {
            "name": "TEST 3: In-Domain Open Clinical Query (No Entity Filter Extracted)",
            "question": "What are the general principles and methods used for blinding and randomization in clinical trials?",
            "filter": None,
            "expected_behavior": "Should recognize no drug/phase constraints (filter=None), retrieve general trial design chunks across the knowledge base, and synthesize principles with page citations.",
            "validate": lambda ans, cits, count, f, q: (
                count > 0 
                and len(cits) > 0 
                and f is None 
                and "Insufficient clinical documentation" not in ans
            )
        },
        {
            "name": "TEST 4: Filter Mismatch Guardrail (Auto-Extracted Phase I Filter)",
            "question": "What are the Phase I dose escalation and starting dose guidelines for Pembrolizumab?",
            "filter": None,
            "expected_behavior": "Should extract Phase I filter. Because knowledge base only contains Phase III data, filter matches 0 chunks and triggers fallback: 'Insufficient clinical documentation provided.'",
            "validate": lambda ans, cits, count, f, q: (
                count == 0 
                and len(cits) == 0 
                and f is not None 
                and f.trial_phase == "Phase I"
                and ans.strip() == "Insufficient clinical documentation provided."
            )
        },
        {
            "name": "TEST 5: Out-of-Domain Guardrail (Cosine Similarity Cutoff)",
            "question": "How do I bake a loaf of sourdough bread with a crispy golden crust from scratch?",
            "filter": None,
            "expected_behavior": "Should fall below similarity score threshold (0.25), retrieve 0 chunks, and return fallback: 'Insufficient clinical documentation provided.'",
            "validate": lambda ans, cits, count, f, q: (
                count == 0 
                and len(cits) == 0 
                and ans.strip() == "Insufficient clinical documentation provided."
            )
        },
        {
            "name": "TEST 6: Information Gap Guardrail (Unrecorded Clinical Fact)",
            "question": "What is the commercial wholesale acquisition cost and retail pricing for Pembrolizumab per 100mg vial in Phase III trials?",
            "filter": None,
            "expected_behavior": "Should adhere to strict operational rule 1 & 2: pricing is not in clinical protocols, so it must return 0 chunks or fallback: 'Insufficient clinical documentation provided.'",
            "validate": lambda ans, cits, count, f, q: (
                ans.strip() == "Insufficient clinical documentation provided."
            )
        }
    ]

    passed_count = 0
    total_tests = len(test_cases)

    for idx, tc in enumerate(test_cases, start=1):
        print(f"\n[{idx}/{total_tests}] {tc['name']}")
        print(f"Question: \"{tc['question']}\"")
        print(f"Filter: {tc.get('filter')}")
        print(f"Expected: {tc['expected_behavior']}")
        print("-" * 50)

        answer, citations, chunk_count, active_filter, search_query = rag.answer_query(
            question=tc["question"],
            explicit_filter=tc.get("filter")
        )

        print(f"-> Cleaned Search Query: {search_query}")
        print(f"-> Active Metadata Filter: {active_filter}")
        print(f"-> Chunks Retrieved: {chunk_count}")
        print(f"-> Citations Generated ({len(citations)}):")
        for c in citations:
            print(f"   * File: {c.source_file} | Page: {c.page} | Phase: {c.trial_phase} | Drug: {c.drug_name}")

        print("\n-> LLM Answer:")
        print(f"{answer}\n")

        # Validation check
        is_passed = tc["validate"](answer, citations, chunk_count, active_filter, search_query)
        if is_passed:
            print(">>> RESULT: [PASS]")
            passed_count += 1
        else:
            print(">>> RESULT: [FAIL]")

        print("=" * 70)

    print(f"\nTEST SUITE SUMMARY: {passed_count}/{total_tests} Tests Passed")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()