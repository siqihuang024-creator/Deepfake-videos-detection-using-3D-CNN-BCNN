"""Before the rented workstation is returned, prove nothing lives only on it.

Two modes, one file, so the pod and the laptop hash the same way.

  On the pod:
      python scripts/pre_teardown_audit.py --audit
  prints what exists only there -- uncommitted and untracked code, runs that
  were never harvested into results/, checkpoints, the detection cache, logs --
  and writes results/diagnostics/pod_inventory.json with a size and SHA-256 for
  every file under results/ and every best.pt under artifacts/.

  On the laptop, after downloading that JSON:
      python scripts/pre_teardown_audit.py --compare results/diagnostics/pod_inventory.json
  checks every one of those files against the local copy -- results/ against
  results/, checkpoints against checkpoints_backup/ -- and reports anything
  missing or different.

Nothing is deleted by either mode.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".json", ".md", ".csv", ".txt")


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git(*args, timeout=120):
    # stderr is kept apart so warnings (line endings, hints) never pass as output.
    try:
        done = subprocess.run(["git"] + list(args), cwd=str(ROOT), timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return "ERROR: timed out"
    if done.returncode != 0:
        return "ERROR: " + done.stderr.decode("utf-8", "replace").strip()
    return done.stdout.decode("utf-8", "replace").strip()


def on_github(rel):
    """True if origin/main already holds this exact file content.

    A fix patched by hand on the pod and later committed from the laptop shows
    up as a local modification here, yet nothing about it would be lost.
    """
    remote = git("rev-parse", "origin/main:" + rel)
    return not remote.startswith("ERROR") and git("hash-object", rel) == remote


def size_of(path):
    total, count = 0, 0
    for base, _, files in os.walk(str(path)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
                count += 1
            except OSError:
                pass
    return total, count


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def audit(output):
    banner("1. CODE")
    print("HEAD        :", git("log", "--oneline", "-1"))
    # The pod's TLS stack has failed over HTTP/2 before; 1.1 is what worked.
    fetched = git("-c", "http.version=HTTP/1.1", "fetch", "-q", timeout=90)
    if fetched.startswith("ERROR"):
        print("fetch failed, origin/main below may be stale:", fetched)
    print("origin/main :", git("log", "--oneline", "-1", "origin/main"))
    ahead = git("log", "--oneline", "origin/main..HEAD")
    print("local commits not on GitHub:", ahead if ahead else "none")
    # results/ is harvested here and downloaded, never committed from the pod;
    # section 4 accounts for it by hash, so it is left out of the git checks.
    changed = git("diff", "--name-only", "HEAD").splitlines()
    changed = [rel for rel in changed if rel and not rel.startswith("results/")]
    same_as_github = [rel for rel in changed if on_github(rel)]
    modified = [rel for rel in changed if rel not in same_as_github]
    print("modified tracked files identical to GitHub (nothing to save):",
          "\n  " + "\n  ".join(same_as_github) if same_as_github else "none")
    print("modified tracked files that DIFFER from GitHub (exist ONLY here):",
          "\n  " + "\n  ".join(modified) if modified else "none")
    if modified:
        diff = git("diff", "origin/main", "--", *modified)
        patch = ROOT / "results" / "diagnostics" / "pod_only_changes.diff"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(diff + "\n", encoding="utf-8")
        print("  full diff against GitHub saved to", patch.relative_to(ROOT).as_posix())
        print(git("diff", "--stat", "origin/main", "--", *modified))
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    untracked = [line for line in untracked if not line.startswith("results/")]
    code_like = [line for line in untracked
                 if line.endswith(CODE_SUFFIXES) and not on_github(line)]
    other = [line for line in untracked if not line.endswith(CODE_SUFFIXES)]
    print("untracked code/config (exists ONLY here):",
          "\n  " + "\n  ".join(code_like) if code_like else "none")
    print("untracked other files outside results/: {}".format(len(other)))
    for line in other[:20]:
        print("  ", line)
    stash = git("stash", "list")
    print("stash:", stash if stash else "none")

    banner("2. RUNS")
    artifacts = ROOT / "artifacts"
    results = ROOT / "results"
    runs = sorted(p for p in artifacts.glob("run_*") if p.is_dir())
    unharvested = []
    print("{:<52}{:>8}{:>9}{:>9}{:>10}".format("run", "history", "harvest", "best.pt", "ckpt size"))
    for run in runs:
        has_history = (run / "logs" / "history.json").exists()
        harvested = (results / run.name / "history.json").exists()
        best = run / "checkpoints" / "best.pt"
        ckpt, _ = size_of(run / "checkpoints")
        if has_history and not harvested:
            unharvested.append(run.name)
        print("{:<52}{:>8}{:>9}{:>9}{:>10}".format(
            run.name[:51], "yes" if has_history else "-",
            "yes" if harvested else ("MISSING" if has_history else "-"),
            "yes" if best.exists() else "-", human(ckpt)))
    print()
    print("runs with a history that results/ does not have:",
          "\n  " + "\n  ".join(unharvested) if unharvested else "none")
    reports = sorted(p.relative_to(ROOT).as_posix() for p in artifacts.glob("run_*/**/report_*.json"))
    unharvested_reports = [r for r in reports
                           if not (results / Path(r).parts[1] / Path(r).name).exists()]
    print("evaluation reports that results/ does not have:",
          "\n  " + "\n  ".join(unharvested_reports) if unharvested_reports else "none")

    banner("3. LARGE STATE")
    total, count = size_of(artifacts / "face_cache")
    print("{:<18} {:>10}  {:,} files".format("detection cache", human(total), count))
    total = sum(size_of(r / "checkpoints")[0] for r in runs)
    print("{:<18} {:>10}".format("all checkpoints", human(total)))
    logs = list(ROOT.glob("*.log")) + list(artifacts.glob("*.log")) \
        + list(artifacts.glob("cache_logs/*.log"))
    print("{:<18} {:>10}  {} files".format(
        "loose logs", human(sum(p.stat().st_size for p in logs)), len(logs)))
    for name, variable in (("DFD", "DFD_ROOT"), ("CelebDFv3", "CELEBDFV3_ROOT")):
        where = os.environ.get(variable)
        if where and Path(where).exists():
            total, count = size_of(where)
            print("{:<18} {:>10}  {:,} files  at {}".format(
                "dataset " + name, human(total), count, where))

    banner("4. HASHING results/ AND CHECKPOINTS")
    inventory = {}
    for path in sorted(results.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            inventory[rel] = {"size": path.stat().st_size, "sha256": sha256(path)}
    checkpoints = {}
    for run in runs:
        best = run / "checkpoints" / "best.pt"
        if best.exists():
            checkpoints[run.name] = {"size": best.stat().st_size, "sha256": sha256(best)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"head": git("rev-parse", "HEAD"), "files": inventory,
                   "checkpoints": checkpoints, "unharvested_runs": unharvested,
                   "unharvested_reports": unharvested_reports,
                   "untracked_code": code_like, "modified": modified,
                   "unpushed": ahead}, handle, indent=1)
    print("{:,} results files and {} checkpoints hashed -> {}".format(
        len(inventory), len(checkpoints), output))
    return 0


def local_checkpoint(run, meta):
    """Where the laptop keeps a pod checkpoint, if it has a matching copy."""
    for path in (ROOT / "checkpoints_backup" / "backup_ckpt" / run / "best.pt",
                 ROOT / "checkpoints_backup" / run / "checkpoints" / "best.pt"):
        if path.exists() and path.stat().st_size == meta["size"] and sha256(path) == meta["sha256"]:
            return path
    return None


def compare(path):
    record = json.load(open(path, encoding="utf-8"))
    files = record["files"]
    missing, different, same = [], [], 0
    for rel, meta in sorted(files.items()):
        local = ROOT / rel
        if not local.exists():
            missing.append(rel)
        elif local.stat().st_size != meta["size"] or sha256(local) != meta["sha256"]:
            different.append(rel)
        else:
            same += 1

    banner("CODE")
    local_head = git("rev-parse", "HEAD")
    # The pod may lag behind; what matters is that it has nothing the laptop lacks.
    contained = not git("merge-base", "--is-ancestor", record["head"], "HEAD").startswith("ERROR")
    print("pod HEAD   :", record["head"])
    print("local HEAD :", local_head,
          "(contains the pod's HEAD)" if contained else "(does NOT contain the pod's HEAD)")
    if record.get("unpushed"):
        print("unpushed commits on the pod:\n" + record["unpushed"])
    for name in record.get("modified", []):
        print("  changed only on the pod:", name)
    for name in record.get("untracked_code", []):
        print("  only on the pod:", name)

    banner("RESULTS")
    print("results/ files on the pod : {:,}".format(len(files)))
    print("  identical locally       : {:,}".format(same))
    print("  missing locally         : {:,}".format(len(missing)))
    print("  present but different   : {:,}".format(len(different)))
    for label, items in (("MISSING", missing), ("DIFFERENT", different)):
        for rel in items[:40]:
            print("   {:<9} {}".format(label, rel))
        if len(items) > 40:
            print("   ... and {} more".format(len(items) - 40))
    for name in record.get("unharvested_runs", []):
        print("  never harvested on the pod:", name)
    for name in record.get("unharvested_reports", []):
        print("  report never harvested on the pod:", name)

    banner("CHECKPOINTS (best.pt)")
    not_backed_up = []
    for run, meta in sorted(record.get("checkpoints", {}).items()):
        found = local_checkpoint(run, meta)
        if found is None:
            not_backed_up.append(run)
        print("  {:<9} {:>9}  {}".format("backed up" if found else "POD ONLY",
                                         human(meta["size"]), run))

    code_clean = contained and not record.get("unpushed") \
        and not record.get("modified") and not record.get("untracked_code")
    results_clean = not missing and not different and not record.get("unharvested_runs") \
        and not record.get("unharvested_reports")
    print()
    print("code     :", "matches" if code_clean else "GAPS ABOVE")
    print("results  :", "matches" if results_clean else "GAPS ABOVE")
    print("best.pt  : {} of {} backed up locally".format(
        len(record.get("checkpoints", {})) - len(not_backed_up), len(record.get("checkpoints", {}))))
    return 0 if code_clean and results_clean else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true", help="Run on the pod.")
    group.add_argument("--compare", metavar="INVENTORY_JSON", help="Run on the laptop.")
    parser.add_argument("--output", default=str(ROOT / "results/diagnostics/pod_inventory.json"))
    args = parser.parse_args()
    if args.audit:
        return audit(Path(args.output))
    return compare(args.compare)


if __name__ == "__main__":
    sys.exit(main())
