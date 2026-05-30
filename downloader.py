import requests
import zipfile
import io
import os
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

def get_recent_failed_run_id(owner: str, repo: str, token: str) -> int:
    """Fetches the most recent failed Actions run ID for a repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.get(url, headers=headers, params={"status": "failure", "per_page": 50})
    response.raise_for_status()
    
    runs = response.json()["workflow_runs"]
    if not runs:
        raise ValueError("No failed runs found.")

    return runs    
    #run = runs[0]
    #print(f"Using run: {run['name']} | id: {run['id']} | created: {run['created_at']}")
    #return run["id"]


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
    Also writes a metadata.json with the run_id and file list.
    """
    run_dir = Path(output_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in logs.items():
        # Filenames from the ZIP can contain slashes (subdirs) — flatten them
        safe_name = filename.replace("/", "_")
        log_path = run_dir / safe_name
        log_path.write_text(content, encoding="utf-8")
        print(f"Saved: {log_path}")

    # Write a metadata sidecar
    metadata = {
        "run_id": run_id,
        "repo": repo,
        "file_count": len(logs),
        "files": list(logs.keys()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nAll logs saved to: {run_dir}")
    return run_dir


token=os.getenv("GITHUB_TOKEN")
runs = get_recent_failed_run_id("python", "cpython", token)
repo = "python/cpython"

for run in runs:
    print(f"Using run: {run['name']} | id: {run['id']} | created: {run['created_at']}")
    run_id = run["id"]
    logs = download_run_logs("python", "cpython", run_id=run_id, token=token)
    save_logs(logs, run_id, repo)

# Print just the test-related steps
#for filename, content in logs.items():
    #if "test" in filename.lower():
    #    print(f"=== {filename} ===")
    #    print(content[:3000])  # first 3000 chars to get the gist