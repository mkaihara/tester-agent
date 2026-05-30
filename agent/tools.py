import os
import requests
from langchain_core.tools import tool

@tool
def get_test_run_history(repo: str, test_name: str, lookback_runs: int = 10) -> dict:
    """
    Retrieves recent CI run results for a specific test name.
    Returns pass/fail history and a flaky probability estimate.
    
    Args:
        repo: GitHub repo in 'owner/name' format
        test_name: the name of the test to look up
        lookback_runs: how many recent runs to check
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    owner, name = repo.split("/")
    runs_url = f"https://api.github.com/repos/{owner}/{name}/actions/runs"
    response = requests.get(
        runs_url,
        headers=headers,
        params={"per_page": lookback_runs, "status": "completed"}
    )
    response.raise_for_status()
    runs = response.json()["workflow_runs"]

    failures = 0
    total = 0
    for run in runs:
        total += 1
        if run["conclusion"] == "failure":
            failures += 1

    flaky_probability = round(failures / total, 2) if total > 0 else 0.0

    return {
        "test_name": test_name,
        "runs_checked": total,
        "failures": failures,
        "passes": total - failures,
        "flaky_probability": flaky_probability,
        "verdict": "flaky" if 0.1 < flaky_probability < 0.9 else "consistent",
    }