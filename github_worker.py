#!/usr/bin/env python3
"""Interactive GitHub sync helper for the RedFM_MLP workspace.

This script intentionally performs Git operations only after interactive
confirmation at the pull and commit/push branch decision points.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TARGET_REMOTE_URL = "https://github.com/ShimingGu/RedFM_MLP.git"
TARGET_REMOTE_DISPLAY = "https://github.com/ShimingGu/RedFM_MLP"
DEFAULT_REMOTE_NAME = "origin"
FALLBACK_REMOTE_NAME = "redfm_mlp"

EXCLUDED_PATHS = (
    "data/clauds/images",
    "data/clauds/catalogs",
)


class GitWorkerError(RuntimeError):
    """Raised for expected git workflow failures."""


def run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and return the completed process."""
    kwargs = {
        "cwd": str(cwd),
        "text": True,
        "check": False,
    }
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

    proc = subprocess.run(args, **kwargs)
    if check and proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        raise GitWorkerError(f"{' '.join(args)} failed: {detail}")
    return proc


def git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture: bool = True,
) -> str:
    """Run a git command and return stripped stdout."""
    proc = run(["git", *args], cwd=cwd, check=check, capture=capture)
    return (proc.stdout or "").strip()


def normalize_remote_url(url: str) -> str:
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def prompt_yes_no(question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_nonempty(question: str) -> str:
    while True:
        answer = input(question).strip()
        if answer:
            return answer
        print("Please enter a non-empty value.")


def ensure_git_repo(start: Path) -> Path:
    proc = run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=False,
    )
    if proc.returncode == 0:
        return Path((proc.stdout or "").strip()).resolve()

    print("No git repository found here. Initializing this folder as a git repository.")
    git(["init"], cwd=start)
    root = Path(git(["rev-parse", "--show-toplevel"], cwd=start)).resolve()

    current = current_branch(root, allow_empty=True)
    if not current:
        git(["checkout", "-B", "main"], cwd=root)
    return root


def current_branch(repo: Path, *, allow_empty: bool = False) -> str:
    branch = git(["branch", "--show-current"], cwd=repo, check=False).strip()
    if branch:
        return branch
    if allow_empty:
        return ""
    raise GitWorkerError("Detached HEAD or no current branch detected; this helper expects a checked-out branch.")


def remotes(repo: Path) -> dict[str, str]:
    output = git(["remote", "-v"], cwd=repo, check=False)
    result: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "(fetch)":
            result[parts[0]] = parts[1]
    return result


def ensure_remote(repo: Path) -> str:
    print(f"Target GitHub repository: {TARGET_REMOTE_DISPLAY}")
    existing = remotes(repo)
    target_norm = normalize_remote_url(TARGET_REMOTE_URL)

    for name, url in existing.items():
        if normalize_remote_url(url) == target_norm:
            print(f"Remote '{name}' already points to the target repository.")
            return name

    if not existing:
        git(["remote", "add", DEFAULT_REMOTE_NAME, TARGET_REMOTE_URL], cwd=repo)
        print(f"Added remote '{DEFAULT_REMOTE_NAME}' -> {TARGET_REMOTE_URL}")
        return DEFAULT_REMOTE_NAME

    origin_url = existing.get(DEFAULT_REMOTE_NAME)
    if origin_url is not None:
        print(f"Remote '{DEFAULT_REMOTE_NAME}' currently points to:")
        print(f"  {origin_url}")
        if prompt_yes_no(f"Replace '{DEFAULT_REMOTE_NAME}' with the target repository?", default=False):
            git(["remote", "set-url", DEFAULT_REMOTE_NAME, TARGET_REMOTE_URL], cwd=repo)
            print(f"Updated remote '{DEFAULT_REMOTE_NAME}' -> {TARGET_REMOTE_URL}")
            return DEFAULT_REMOTE_NAME

    remote_name = FALLBACK_REMOTE_NAME
    while remote_name in existing:
        remote_name = prompt_nonempty(
            f"Remote name '{remote_name}' already exists. Enter a new remote name for the target repository: "
        )
    git(["remote", "add", remote_name, TARGET_REMOTE_URL], cwd=repo)
    print(f"Added remote '{remote_name}' -> {TARGET_REMOTE_URL}")
    return remote_name


def remote_branch_exists(repo: Path, remote: str, branch: str) -> bool:
    proc = run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/{remote}/{branch}"],
        cwd=repo,
        check=False,
        capture=False,
    )
    return proc.returncode == 0


def fetch_remote(repo: Path, remote: str) -> None:
    print(f"Fetching '{remote}'...")
    git(["fetch", "--prune", remote], cwd=repo, capture=False)


def ahead_behind(repo: Path, remote: str, branch: str) -> tuple[int, int] | None:
    if not remote_branch_exists(repo, remote, branch):
        return None
    output = git(
        ["rev-list", "--left-right", "--count", f"HEAD...{remote}/{branch}"],
        cwd=repo,
    )
    ahead_text, behind_text = output.split()
    return int(ahead_text), int(behind_text)


def report_remote_status(repo: Path, remote: str, branch: str) -> None:
    counts = ahead_behind(repo, remote, branch)
    if counts is None:
        print(f"Remote branch '{remote}/{branch}' does not exist yet.")
        return

    ahead, behind = counts
    if behind:
        print(f"GitHub has {behind} commit(s) on '{remote}/{branch}' that are not present locally.")
        log_output = git(["log", "--oneline", "--decorate", "-n", "10", f"HEAD..{remote}/{branch}"], cwd=repo)
        if log_output:
            print("Remote-only commits:")
            print(log_output)
    else:
        print(f"No new commits found on GitHub branch '{remote}/{branch}'.")

    if ahead:
        print(f"Local branch has {ahead} commit(s) not present on '{remote}/{branch}'.")


def maybe_pull(repo: Path, remote: str, current: str) -> None:
    counts = ahead_behind(repo, remote, current)
    default_pull = False
    if counts is not None:
        _, behind = counts
        if behind > 0:
            print(f"\nWARNING: Your local branch is behind the remote by {behind} commit(s).")
            print("Pushes will be rejected unless you pull remote changes first.")
            default_pull = True

    if not prompt_yes_no("Do you want to pull remote changes before staging?", default=default_pull):
        print("Skipping pull.")
        return

    pull_branch = input(f"Pull from which branch? Press Enter for current branch '{current}': ").strip() or current
    if not remote_branch_exists(repo, remote, pull_branch):
        raise GitWorkerError(f"Remote branch '{remote}/{pull_branch}' does not exist. Aborting before pull.")

    selected_counts = ahead_behind(repo, remote, pull_branch)
    if selected_counts is not None:
        ahead, behind = selected_counts
        print(f"Before pull against '{remote}/{pull_branch}': local ahead={ahead}, remote ahead={behind}")

    print(f"Pulling from '{remote}/{pull_branch}' into current local branch '{current}'...")
    git(["pull", remote, pull_branch], cwd=repo, capture=False)


def is_ignored(repo: Path, path: str) -> bool:
    proc = run(
        ["git", "check-ignore", "-q", path],
        cwd=repo,
        check=False,
        capture=False,
    )
    return proc.returncode == 0


def stage_everything_except_large_data(repo: Path) -> None:
    pathspecs = ["."]
    for path in EXCLUDED_PATHS:
        if is_ignored(repo, path):
            continue
        pathspecs.append(f":(exclude){path}")
        pathspecs.append(f":(exclude){path}/**")

    print("Staging all changes except large CLAUDS images and catalogues:")
    for path in EXCLUDED_PATHS:
        if is_ignored(repo, path):
            print(f"  excluding {path}/ (via .gitignore)")
        else:
            print(f"  excluding {path}/")
    git(["add", "-A", "--", *pathspecs], cwd=repo)


def has_staged_changes(repo: Path) -> bool:
    proc = run(["git", "diff", "--cached", "--quiet"], cwd=repo, check=False, capture=False)
    return proc.returncode != 0


def commit_staged_changes(repo: Path) -> None:
    staged = git(["diff", "--cached", "--name-status"], cwd=repo, check=False)
    if staged:
        print("Staged changes:")
        print(staged)

    if not has_staged_changes(repo):
        print("No staged changes to commit. Skipping commit step.")
        return

    message = prompt_nonempty("Enter git commit message: ")
    git(["commit", "-m", message], cwd=repo)
    print("Commit created.")


def choose_push_branch(current: str) -> str:
    return input(f"Push to which branch? Press Enter for current branch '{current}': ").strip() or current


def push(repo: Path, remote: str, current: str, push_branch: str) -> None:
    if push_branch == current:
        print(f"Pushing current branch '{current}' to '{remote}/{push_branch}'...")
        git(["push", remote, current], cwd=repo, capture=False)
    else:
        print(f"Pushing local HEAD to '{remote}/{push_branch}'...")
        git(["push", remote, f"HEAD:{push_branch}"], cwd=repo, capture=False)


def print_step(number: int, title: str) -> None:
    print()
    print(f"Step {number}/7: {title}")
    print("-" * (len(title) + 10))


def main() -> int:
    try:
        start = Path.cwd().resolve()

        print_step(1, "Check or establish GitHub remote connection")
        repo = ensure_git_repo(start)
        print(f"Repository root: {repo}")
        remote = ensure_remote(repo)

        current = current_branch(repo)
        print(f"Current checked-out branch: {current}")

        print_step(2, "Fetch GitHub and report remote-only commits")
        fetch_remote(repo, remote)
        report_remote_status(repo, remote, current)

        print_step(3, "Choose whether and where to pull")
        maybe_pull(repo, remote, current)
        current = current_branch(repo)

        print_step(4, "Stage everything except large CLAUDS data")
        stage_everything_except_large_data(repo)

        print_step(5, "Enter commit message and commit")
        commit_staged_changes(repo)

        print_step(6, "Choose push branch")
        current = current_branch(repo)
        push_branch = choose_push_branch(current)

        print_step(7, "Push to GitHub")
        push(repo, remote, current, push_branch)
        print("GitHub worker finished successfully.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except GitWorkerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
