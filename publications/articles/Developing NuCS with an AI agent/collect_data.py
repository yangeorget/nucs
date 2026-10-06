"""
Collects the data of charts 1 and 2 of the article from the git history of NuCS.

Usage, from the root of the repository:

    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" commits
    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" tests [2026-03 ...]
    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" coverage 2026-03 2026-04 ...

- commits: the commits of each month, all and with the Co-Authored-By: Claude trailer (chart 1).
- tests: at the last commit of each month, the test functions in tests/ and the tests that pytest collects (chart 2).
- coverage: at the last commit of each given month, the line coverage of nucs/ by the tests without the JIT, as the CI
  measures it (chart 2). Each month runs in its own git worktree, in parallel, and takes 20 to 50 minutes.

The results go to data/*.csv next to this script. Each row records the Python and numba versions that measured it:
an old commit may need the dependencies of its time (run the script with the Python of a venv that has them). With
months, tests measures only these months; coverage measures only the months that are not in coverage_per_month.csv.
"""

import csv
import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
DATA = Path(__file__).resolve().parent / "data"
PYTHON = sys.executable
ENV = f"py{sys.version_info.major}.{sys.version_info.minor} numba{importlib.metadata.version('numba')}"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def month_ends() -> list[tuple[str, str]]:
    """The last commit on the first-parent history of main for each month, oldest first."""
    last: dict[str, str] = {}
    for line in git("log", "--first-parent", "--format=%H %ad", "--date=format:%Y-%m", "main").splitlines():
        sha, month = line.split()
        last.setdefault(month, sha)  # the log is newest first, so the first one seen is the last of the month
    return sorted((month, sha) for month, sha in last.items())


def read_csv(name: str) -> dict[str, list[object]]:
    """The rows of a CSV of data/, by month."""
    if not (DATA / name).exists():
        return {}
    with open(DATA / name) as f:
        return {row[0]: list(row) for row in csv.reader(f) if row[0] != "month"}


def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {DATA / name} ({len(rows)} rows)")


def commits() -> None:
    all_nb: Counter[str] = Counter()
    agent_nb: Counter[str] = Counter()
    log = git("log", "--no-merges", "--format=%ad%x09%(trailers:key=Co-Authored-By,valueonly,separator=;)",
              "--date=format:%Y-%m", "main")
    for line in log.splitlines():
        month, _, trailers = line.partition("\t")
        all_nb[month] += 1
        if "Claude" in trailers:
            agent_nb[month] += 1
    # Lines added + deleted, by kind of file. The data files (datasets/, *.fzn, *.dzn, ...) are left out.
    churn: dict[str, Counter[str]] = {}
    month = ""
    for line in git("log", "--no-merges", "--numstat", "--format=@%ad", "--date=format:%Y-%m", "main").splitlines():
        if line.startswith("@"):
            month = line[1:]
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] == "-":  # a blank line or a binary file
            continue
        changed, path = int(parts[0]) + int(parts[1]), parts[2]
        kind = ("code" if path.startswith(("nucs/", "ncs/")) and path.endswith(".py")  # ncs/ until 2024-08-22
                else "tests" if path.startswith("tests/")
                else "docs" if path.endswith((".md", ".rst")) and not path.startswith("datasets/")
                else "")
        if kind:
            churn.setdefault(month, Counter())[kind] += changed
    rows = [[m, all_nb[m], agent_nb[m], all_nb[m] - agent_nb[m],
             churn.get(m, Counter())["code"], churn.get(m, Counter())["tests"], churn.get(m, Counter())["docs"]]
            for m in sorted(all_nb)]
    write_csv("commits_per_month.csv", ["month", "commits", "with_trailer", "without_trailer",
                                        "lines_changed_code", "lines_changed_tests", "lines_changed_docs"], rows)


def worktree(sha: str, root: str) -> Path:
    path = Path(root) / sha[:10]
    git("worktree", "add", "--detach", str(path), sha)
    return path


def drop_worktree(path: Path) -> None:
    git("worktree", "remove", "--force", str(path))


def env_for(path: Path) -> dict[str, str]:
    # PYTHONPATH makes the worktree win over the editable install of the repository (design/benchmarking.md, rule 2).
    return {**os.environ, "PYTHONPATH": str(path), "NUMBA_DISABLE_JIT": "1"}


def tests(months: list[str]) -> None:
    done = read_csv("tests_per_month.csv")
    with tempfile.TemporaryDirectory() as root:
        for month, sha in month_ends():
            if months and month not in months:
                continue
            listing = git("ls-tree", "-r", "--name-only", sha, "--", "tests").splitlines()
            files = [f for f in listing if f.endswith(".py")]
            functions = sum(len(re.findall(r"^\s*def test_", git("show", f"{sha}:{f}"), re.M)) for f in files)
            path = worktree(sha, root)
            try:
                out = subprocess.run([PYTHON, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                                     cwd=path, env=env_for(path), capture_output=True, text=True).stdout
                match = re.search(r"(\d+) tests? collected(?:.*?(\d+) errors?)?", out)
                if match:
                    collected, errors = int(match[1]), int(match[2] or 0)
                else:
                    collected, errors = (0, 0) if "no tests collected" in out else ("", "")
            finally:
                drop_worktree(path)
            done[month] = [month, sha[:7], len(files), functions, collected, errors, ENV]
            print(done[month], flush=True)
    write_csv("tests_per_month.csv",
              ["month", "commit", "test_files", "test_functions", "collected", "collect_errors", "env"],
              [done[m] for m in sorted(done)])


def measure_coverage(month: str, sha: str, root: str) -> list[object]:
    path = worktree(sha, root)
    try:
        env = env_for(path)
        run = subprocess.run([PYTHON, "-m", "coverage", "run", "--source=nucs", "-m", "pytest", "-q",
                              "-p", "no:cacheprovider"], cwd=path, env=env, capture_output=True, text=True)
        summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-200:]
        counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", summary)}
        report = subprocess.run([PYTHON, "-m", "coverage", "report"], cwd=path, env=env, capture_output=True, text=True)
        total = next((line.split() for line in report.stdout.splitlines() if line.startswith("TOTAL")), None)
        statements, missed, percent = (int(total[1]), int(total[2]), total[-1].rstrip("%")) if total else ("", "", "")
    finally:
        drop_worktree(path)
    row = [month, sha[:7], statements, missed, percent, counts.get("passed", 0), counts.get("failed", 0),
           counts.get("error", 0) + counts.get("errors", 0), counts.get("skipped", 0), ENV]
    print(row, flush=True)
    return row


def coverage(months: list[str]) -> None:
    header = ["month", "commit", "statements", "missed", "percent", "passed", "failed", "errors", "skipped", "env"]
    done = read_csv("coverage_per_month.csv")
    todo = [(m, sha) for m, sha in month_ends() if m in months and m not in done]
    with tempfile.TemporaryDirectory() as root, ThreadPoolExecutor(max_workers=len(todo) or 1) as pool:
        for row in pool.map(lambda m_sha: measure_coverage(*m_sha, root), todo):
            done[str(row[0])] = row
            write_csv("coverage_per_month.csv", header, [done[m] for m in sorted(done)])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "commits":
        commits()
    elif command == "tests":
        tests(sys.argv[2:])
    elif command == "coverage":
        coverage(sys.argv[2:])
    else:
        sys.exit(__doc__)
