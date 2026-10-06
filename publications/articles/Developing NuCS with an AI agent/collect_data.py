"""
Collects the data of charts 1, 2 and 3 of the article from the git history of NuCS.

Usage, from the root of the repository:

    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" commits
    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" tests [2026-03 ...]
    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" coverage 2026-03 2026-04 ...
    python "publications/articles/Developing NuCS with an AI agent/collect_data.py" speed [--repeats 3] [v18.0.0 ...]

- commits: the commits of each month, all and with the Co-Authored-By: Claude trailer (chart 1).
- tests: at the last commit of each month, the test functions in tests/ and the tests that pytest collects (chart 2).
- coverage: at the last commit of each given month, the line coverage of nucs/ by the tests without the JIT, as the CI
  measures it (chart 2). Each month runs in its own git worktree, in parallel, and takes 20 to 50 minutes.
- speed: for each release tag of SPEED_TAGS, the time to solve the problems of speed_driver.py (chart 3). Each tag has
  its own worktree and Numba cache in tmp/, and a Python 3.12 venv with the numba and numpy pins of the tag. The tags
  run one after the other, never in parallel, in an order that alternates at each repetition (design/benchmarking.md).
  Each run goes to speed_runs.csv, with its statistics, so that a change of the search tree is visible.

The results go to data/*.csv next to this script. Each row records the Python and numba versions that measured it:
an old commit may need the dependencies of its time (run the script with the Python of a venv that has them). With
months, tests measures only these months; coverage measures only the months that are not in coverage_per_month.csv.
"""

import csv
import hashlib
import importlib.metadata
import json
import tomllib
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


SPEED_TAGS = ["v4.8.1", "v6.0.0", "v9.0.0", "v9.1.3", "v10.1.0", "v11.2.0", "v12.4.9", "v14.1.0", "v15.0.0",
              "v16.1.0", "v17.1.0", "v18.0.0"]
SPEED_PROBLEMS = {  # the problem, and the file that must exist in the tag
    "queens_12": "nucs/examples/queens/queens_problem.py",
    "all_interval_13": "nucs/examples/all_interval_series/all_interval_series_problem.py",
    "bibd_10": "nucs/examples/bibd/bibd_problem.py",
}
SPEED_PYTHON = Path.home() / ".local/bin/python3.12"
SPEED_ROOT = REPO / "tmp" / "article-speed"
SPEED_DRIVER = Path(__file__).resolve().parent / "speed_driver.py"
# Kept from the statistics of a run: the tree and the propagation, to see a change of the search between two tags.
SPEED_STATS = ["SOLVER_BACKTRACK_NB", "SOLVER_CHOICE_NB", "ALG_BC_NB", "PROPAGATOR_FILTER_NB"]


def speed_venv(tag: str) -> Path:
    """A Python 3.12 venv with the dependencies of the tag, shared by the tags that have the same pins."""
    deps = tomllib.loads(git("show", f"{tag}:pyproject.toml"))["project"]["dependencies"] + ["enlighten"]
    path = SPEED_ROOT / ("venv-" + hashlib.sha1(" ".join(sorted(deps)).encode()).hexdigest()[:8])
    if not (path / "bin" / "python").exists():
        subprocess.run(["uv", "venv", "-q", "--python", str(SPEED_PYTHON), str(path)], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(path / "bin" / "python"), *deps], check=True)
    return path / "bin" / "python"


def speed_worktree(tag: str) -> Path:
    path = SPEED_ROOT / tag
    if not path.exists():
        git("worktree", "add", "--detach", str(path), tag)
    return path


def speed(args: list[str]) -> None:
    repeats = 3
    if args[:1] == ["--repeats"]:
        repeats, args = int(args[1]), args[2:]
    tags = args or SPEED_TAGS
    setups = {tag: (speed_worktree(tag), speed_venv(tag)) for tag in tags}
    rows = []
    for repeat in range(repeats):
        for tag in tags if repeat % 2 == 0 else list(reversed(tags)):
            worktree, python = setups[tag]
            env = {**os.environ, "PYTHONPATH": str(worktree), "NUMBA_CACHE_DIR": str(worktree / ".numba-cache")}
            env.pop("NUMBA_DISABLE_JIT", None)
            for problem, required in SPEED_PROBLEMS.items():
                if not (worktree / required).exists():
                    continue
                run = subprocess.run([str(python), str(SPEED_DRIVER), problem], cwd=worktree, env=env,
                                     capture_output=True, text=True, timeout=1800)
                line = next((x for x in run.stdout.splitlines() if x.startswith("RESULT ")), None)
                if line is None:
                    print(f"{tag} {problem}: no result\n{run.stderr[-500:]}", flush=True)
                    continue
                result = json.loads(line[len("RESULT "):])
                # The worktree must be the NuCS that ran (design/benchmarking.md, rule 2).
                assert result["nucs_file"].startswith(str(worktree)), result["nucs_file"]
                numba = subprocess.run([str(python), "-c", "import numba; print(numba.__version__)"],
                                       capture_output=True, text=True).stdout.strip()
                date = git("log", "-1", "--format=%ad", "--date=short", tag).strip()
                stats = [result["statistics"].get(k, "") for k in SPEED_STATS]
                rows.append([tag, date, problem, repeat, result["time_ms"], result["solutions"], *stats,
                             f"py{result['python']} numba{numba}"])
                print(rows[-1], flush=True)
    header = ["tag", "date", "problem", "repeat", "time_ms", "solutions", *[k.lower() for k in SPEED_STATS], "env"]
    write_csv("speed_runs.csv" if not args else "speed_runs_partial.csv", header, rows)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "commits":
        commits()
    elif command == "tests":
        tests(sys.argv[2:])
    elif command == "coverage":
        coverage(sys.argv[2:])
    elif command == "speed":
        speed(sys.argv[2:])
    else:
        sys.exit(__doc__)
