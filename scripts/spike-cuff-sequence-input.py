import argparse
import hashlib
import json
from pathlib import Path

from solver_cuff_sequence import combine_cuff_controls
from solver_process_budget import read_regular


parser = argparse.ArgumentParser(description="Combine source cuff closure and allowance folding; not turning")
parser.add_argument("--sewing-input", type=Path, required=True)
parser.add_argument("--fold-input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--crease-frame-region", choices=("body", "allowance"),
                    help="Explicit facing material-frame side at outer-crease anchors; required when both sides contain the anchor")
args = parser.parse_args()
inputs = [read_regular(path, 50 * 1024 ** 2) for path in (args.sewing_input, args.fold_input)]
control = combine_cuff_controls(*(json.loads(content) for content in inputs),
                                crease_frame_region=args.crease_frame_region)
control["provenance"]["inputSha256"] = {name: hashlib.sha256(content).hexdigest()
    for name, content in zip(("sewing", "fold"), inputs)}
encoded = (json.dumps(control, indent=2, allow_nan=False) + "\n").encode()
output = args.output.resolve()
output.mkdir(mode=0o700)
(output / "canonical.json").write_bytes(encoded)
(output / "placement.json").write_text(json.dumps({"placedMeters": control["placedMeters"],
    "canonicalDigest": hashlib.sha256(encoded).hexdigest()}, allow_nan=False) + "\n")
print(output)
