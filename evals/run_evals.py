import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.graph import cricket_graph
from backend.schemas import MatchState, ExplanationMode


def load_scenarios():
    """Load eval scenarios from JSON."""
    path = os.path.join(os.path.dirname(__file__), "scenarios.json")
    with open(path, "r") as f:
        return json.load(f)


def run_single_eval(scenario: dict) -> dict:
    """
    Runs one scenario through the full graph and evaluates:
    - Did we get the right query type?
    - Did the recommendation contain expected keywords?
    - Was confidence above minimum threshold?
    - Did all agents run without error?
    """
    print(f"\nRunning eval: {scenario['id']} — {scenario['description']}")

    # Build match state
    try:
        match_state = MatchState(**scenario["match_state"])
    except Exception as e:
        return {
            "id": scenario["id"],
            "passed": False,
            "error": f"Invalid match state: {e}",
            "checks": {}
        }

    # Build initial graph state
    initial_state = {
        "user_query": scenario["query"],
        "match_id": scenario["match_state"]["match_id"],
        "explanation_mode": ExplanationMode.ANALYST,
        "match_state": match_state,
        "metrics": None,
        "query_type": None,
        "game_state_summary": None,
        "strategy_analysis": None,
        "historical_context": None,
        "simulation_result": None,
        "prediction_result": None,
        "critic_approved": None,
        "critic_feedback": None,
        "retry_count": 0,
        "final_recommendation": None,
        "final_reasoning": None,
        "final_confidence": None,
        "uncertainty_reason": None,
        "agent_traces": []
    }

    start_time = time.time()

    try:
        result = cricket_graph.invoke(initial_state)
        elapsed = round(time.time() - start_time, 2)
    except Exception as e:
        return {
            "id": scenario["id"],
            "passed": False,
            "error": str(e),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "checks": {}
        }

    # ── Evaluate results ──────────────────────────────────────────────
    checks = {}

    # Check 1: Correct query type routing
    actual_qt = result.get("query_type")
    actual_qt_val = actual_qt.value if hasattr(actual_qt, "value") else str(actual_qt)
    checks["correct_routing"] = (
        actual_qt_val == scenario["expected_query_type"]
    )

    # Check 2: Recommendation not empty
    recommendation = result.get("final_recommendation", "")
    checks["has_recommendation"] = (
        bool(recommendation) and
        recommendation != "No recommendation generated"
    )

    # Check 3: Expected keywords present
    rec_lower = recommendation.lower()
    keywords_found = [
        kw for kw in scenario["expected_keywords"]
        if kw.lower() in rec_lower
    ]
    keyword_hit_rate = len(keywords_found) / len(scenario["expected_keywords"])
    checks["keyword_match"] = keyword_hit_rate >= 0.25  # 25% keyword hit rate
    checks["keyword_hit_rate"] = round(keyword_hit_rate * 100, 1)
    checks["keywords_found"] = keywords_found

    # Check 4: Confidence above minimum
    confidence = result.get("final_confidence", 0.0)
    checks["confidence_ok"] = confidence >= scenario["expected_confidence_min"]
    checks["actual_confidence"] = round(confidence * 100, 1)

    # Check 5: Agent traces populated
    traces = result.get("agent_traces", [])
    checks["has_traces"] = len(traces) >= 2
    checks["trace_count"] = len(traces)

    # Check 6: No critic rejection (or resolved within retries)
    checks["critic_approved"] = result.get("critic_approved", True)

    # Overall pass/fail
    core_checks = [
        checks["correct_routing"],
        checks["has_recommendation"],
        checks["keyword_match"],
        checks["confidence_ok"],
        checks["has_traces"]
    ]
    passed = all(core_checks)

    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "query": scenario["query"],
        "passed": passed,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "recommendation_preview": recommendation[:120] + "..." if len(recommendation) > 120 else recommendation,
        "error": None
    }


def run_all_evals():
    """
    Runs all eval scenarios and prints a summary report.
    This is your measurable performance metric.
    """
    scenarios = load_scenarios()
    results = []

    print("=" * 60)
    print("CRICKET AI — EVALUATION SUITE")
    print(f"Running {len(scenarios)} scenarios...")
    print("=" * 60)

    import time as time_module
    for i,scenario in enumerate(scenarios):
        result = run_single_eval(scenario)
        results.append(result)
        #pause between scenarios to avoid rate limits or overloading
        if i<len(scenarios)-1:
            time_module.sleep(3) # 3 second pause between scenarios

        # Print immediate result
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} | {result['id']} | {result.get('elapsed_seconds', 0)}s")

        if not result["passed"]:
            checks = result.get("checks", {})
            if not checks.get("correct_routing"):
                print(f"       ↳ Wrong routing")
            if not checks.get("has_recommendation"):
                print(f"       ↳ Empty recommendation")
            if not checks.get("keyword_match"):
                print(
                    f"       ↳ Low keyword match: "
                    f"{checks.get('keyword_hit_rate', 0)}% "
                    f"(found: {checks.get('keywords_found', [])})"
                )
            if not checks.get("confidence_ok"):
                print(
                    f"       ↳ Low confidence: "
                    f"{checks.get('actual_confidence', 0)}%"
                )
        else:
            checks = result.get("checks", {})
            print(
                f"       ↳ Routing: ✓ | "
                f"Keywords: {checks.get('keyword_hit_rate', 0)}% | "
                f"Confidence: {checks.get('actual_confidence', 0)}% | "
                f"Traces: {checks.get('trace_count', 0)}"
            )

    # ── Summary report ────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    accuracy = round((passed / total) * 100, 1)

    avg_elapsed = round(
        sum(r.get("elapsed_seconds", 0) for r in results) / total, 2
    )

    routing_correct = sum(
        1 for r in results
        if r.get("checks", {}).get("correct_routing")
    )
    avg_confidence = round(
        sum(
            r.get("checks", {}).get("actual_confidence", 0)
            for r in results
        ) / total, 1
    )
    avg_keywords = round(
        sum(
            r.get("checks", {}).get("keyword_hit_rate", 0)
            for r in results
        ) / total, 1
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total scenarios  : {total}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"Overall accuracy : {accuracy}%")
    print(f"Routing accuracy : {routing_correct}/{total} ({round(routing_correct/total*100, 1)}%)")
    print(f"Avg confidence   : {avg_confidence}%")
    print(f"Avg keyword match: {avg_keywords}%")
    print(f"Avg response time: {avg_elapsed}s")
    print("=" * 60)

    # Save results to file
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "accuracy": accuracy,
                "routing_accuracy": round(routing_correct / total * 100, 1),
                "avg_confidence": avg_confidence,
                "avg_keyword_match": avg_keywords,
                "avg_response_time": avg_elapsed
            },
            "results": results
        }, f, indent=2)

    print(f"\nDetailed results saved to: evals/eval_results.json")
    return accuracy


if __name__ == "__main__":
    run_all_evals()
