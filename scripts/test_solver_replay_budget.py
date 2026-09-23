"""Exercise the real replay CLI budget block without importing its solvers."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import resource
import subprocess
import sys
import unittest
from unittest.mock import patch


REPLAY = Path(__file__).with_name("replay-rest-filtered-continuation.py")


def budget_block():
    tree = ast.parse(REPLAY.read_text(), filename=str(REPLAY))
    assignments = {node.targets[0].id: index for index, node in enumerate(tree.body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    # Extract actual argument validation and resource admission, ending before
    # the first captured report read. The numerical body is never executed.
    selected = tree.body[assignments["parser"]:assignments["report"]]
    return ast.Module(body=selected, type_ignores=[])


class ReplayBudgetTests(unittest.TestCase):
    def execute_arguments(self, arguments):
        with patch.object(sys, "argv", [str(REPLAY), "unused-run", *arguments]), \
                patch.object(resource, "setrlimit") as limits:
            namespace = {"argparse": argparse, "Path": Path, "resource": resource}
            exec(compile(budget_block(), str(REPLAY), "exec"), namespace)
            return namespace["replay_arguments"], limits.call_args_list

    def test_default_and_explicit_bounded_cpu_limits(self):
        for arguments, expected in (([], 100), (["--cpu-limit-seconds", "1"], 1),
                                    (["--cpu-limit-seconds", "600"], 600),
                                    (["--cpu-limit-seconds", "3600"], 3600)):
            with self.subTest(arguments=arguments):
                parsed, calls = self.execute_arguments(arguments)
                self.assertEqual(parsed.cpu_limit_seconds, expected)
                self.assertEqual([call.args for call in calls], [
                    (resource.RLIMIT_CPU, (expected, expected + 5)),
                    (resource.RLIMIT_CORE, (0, 0))])

    def test_invalid_budget_rejects_before_resource_or_artifact_access(self):
        for value in ("0", "-1", "3601", "1.5", "nan", "inf"):
            with self.subTest(value=value), patch.object(sys, "argv", [
                    str(REPLAY), "unused-run", "--cpu-limit-seconds", value]), \
                    patch.object(resource, "setrlimit") as limits, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                exec(compile(budget_block(), str(REPLAY), "exec"),
                     {"argparse": argparse, "Path": Path, "resource": resource})
            self.assertEqual(failure.exception.code, 2)
            limits.assert_not_called()

    def test_native_child_process_receives_the_requested_limits(self):
        # No CPU limit is installed in the unittest process itself.
        program = "import argparse, resource\nfrom pathlib import Path\n" + ast.unparse(budget_block())
        program += "\nprint(resource.getrlimit(resource.RLIMIT_CPU), resource.getrlimit(resource.RLIMIT_CORE))\n"
        result = subprocess.run([sys.executable, "-c", program, "unused-run", "--cpu-limit-seconds", "600"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "(600, 605) (0, 0)")


if __name__ == "__main__":
    unittest.main()
