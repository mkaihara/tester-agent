import requests
import zipfile
import io
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_recent_failed_runs(owner: str, repo: str, token: str) -> list:
    """Fetches the most recent failed Actions runs for a repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.get(url, headers=headers, params={"status": "failure", "per_page": 100})
    response.raise_for_status()

    runs = response.json()["workflow_runs"]
    if not runs:
        raise ValueError("No failed runs found.")
    return runs


def download_run_logs(owner: str, repo: str, run_id: int, token: str) -> dict[str, str]:
    """
    Downloads and extracts all log files for a GitHub Actions run.
    Returns a dict of {filename: log_content}.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }

    response = requests.get(url, headers=headers, allow_redirects=True)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    logs = {}
    for name in zip_file.namelist():
        with zip_file.open(name) as f:
            logs[name] = f.read().decode("utf-8", errors="replace")

    return logs

def save_logs(logs: dict[str, str], run_id: int, repo: str, output_dir: str = "fixtures/raw") -> Path:
    """
    Saves each log file to fixtures/raw/{run_id}/.
    Also writes a metadata.json with the run_id, repo, and file list.
    """
    run_dir = Path(output_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in logs.items():
        safe_name = filename.replace("/", "_")
        log_path = run_dir / safe_name
        log_path.write_text(content, encoding="utf-8")
        print(f"Saved: {log_path}")

    metadata = {
        "run_id": run_id,
        "repo": repo,
        "file_count": len(logs),
        "files": list(logs.keys()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nAll logs saved to: {run_dir}")
    return run_dir

if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    repo = "python/cpython"

    runs = get_recent_failed_runs("python", "cpython", token)
    for run in runs:
        print(f"Using run: {run['name']} | id: {run['id']} | created: {run['created_at']}")
        logs = download_run_logs("python", "cpython", run_id=run["id"], token=token)
        save_logs(logs, run["id"], repo)
