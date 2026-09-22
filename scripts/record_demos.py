"""Record scripted-expert demonstrations. Thin wrapper over the bench's
`scripts/record_dataset.py` so the dataset format is defined in one place.

    python scripts/record_demos.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --tasks lift:red --episodes 200

Every argument is forwarded. The bench script keeps successful expert
episodes only and samples instructions from the training templates.
"""
import os
import subprocess
import sys

import so_arm100_sim


def main():
    bench_root = os.path.dirname(os.path.dirname(os.path.abspath(so_arm100_sim.__file__)))
    script = os.path.join(bench_root, "scripts", "record_dataset.py")
    if not os.path.exists(script):
        sys.exit("the bench's scripts/record_dataset.py was not found next to the installed "
                 "so_arm100_sim package; install the bench from a source checkout (pip install -e)")
    sys.exit(subprocess.call([sys.executable, script] + sys.argv[1:]))


if __name__ == "__main__":
    main()
