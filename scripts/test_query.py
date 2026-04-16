"""
scripts/test_query.py — End-to-end API smoke tests.

Usage:
    python scripts/test_query.py
    python scripts/test_query.py --url https://your-app.onrender.com
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import requests
except ImportError:
    print("pip install requests"); sys.exit(1)

TEST_CASES = [
    {"query": "How do I configure an existing Control-M job?",        "expect_category": "Configuration",    "expect_wu": "IMS_WU_7036"},
    {"query": "Create a new Control-M job for a new project",          "expect_category": "Provisioning",     "expect_wu": "PMS_WU_2212"},
    {"query": "Migrate a Control-M job to another server",             "expect_category": "Migration",        "expect_wu": "PMS_WU_7003"},
    {"query": "Decommission and delete an existing Control-M job",     "expect_category": "Decommissioning",  "expect_wu": "IMS_WU_7038"},
]

def run(base_url):
    print(f"\n🧪 Smoke tests → {base_url}\n{'─'*60}")
    r = requests.get(f"{base_url}/health", timeout=10)
    h = r.json()
    print(f"✅ Health OK — {h['vector_count']} vectors | LLM: {h['llm_model']}\n")

    passed = failed = 0
    for tc in TEST_CASES:
        r = requests.post(f"{base_url}/query",
            json={"query": tc["query"], "top_k": 3}, timeout=30)
        data = r.json()
        wu_match  = tc["expect_wu"] in data.get("wu_ids", [])
        has_answer = len(data.get("answer", "")) > 20
        ok = wu_match and has_answer
        status = "✅" if ok else "❌"
        print(f"{status} WU match={wu_match} | answer={'yes' if has_answer else 'no'}")
        print(f"   Query  : {tc['query'][:60]}")
        print(f"   WU Ids : {data.get('wu_ids', [])}")
        print(f"   Answer : {data.get('answer','')[:120]}…\n")
        passed += ok; failed += not ok

    print(f"{'─'*60}\nResults: {passed} passed, {failed} failed\n")
    return failed == 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    args = p.parse_args()
    sys.exit(0 if run(args.url) else 1)
