import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def group_members(group):
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("spike-3d-solver.py")
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "HOME": str(args.output.resolve())}
    command = [args.python, str(script), "--fixture", "equal-seam", "--steps", "1000", "--output", str(args.output / "cancelled")]
    with (args.output / "cancelled.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log, start_new_session=True)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError("Fixture ended before cancellation could be tested")
        members = group_members(process.pid)
        started = time.monotonic()
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        remaining = group_members(process.pid)
        termination_seconds = time.monotonic() - started
        if remaining:
            os.killpg(process.pid, signal.SIGKILL)
            raise RuntimeError(f"Process group survived cancellation: {remaining}")
    with (args.output / "restart.log").open("w") as log:
        restarted = subprocess.run([args.python, str(script), "--fixture", "equal-seam", "--output", str(args.output / "restarted")], env=environment, stdout=log, stderr=log, timeout=300, check=True)
    report = {"observedGroupMembersBeforeCancel": len(members), "remainingGroupMembers": len(remaining), "cancelExitCode": process.returncode, "terminationSeconds": termination_seconds, "restartExitCode": restarted.returncode, "restartReportPresent": (args.output / "restarted/report.json").is_file(), "scope": "Observed solver process group; not an application lease/install race test"}
    (args.output / "lifecycle-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
