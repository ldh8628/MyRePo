"""
FoundPose pipeline runner.
Injects necessary paths before executing pipeline scripts.
"""
import sys
import os

# Inject paths (required because PYTHONPATH may not work correctly on Windows)
REPO = "D:/LDH_ws/foundpose-main"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/external/bop_toolkit")
sys.path.insert(0, f"{REPO}/external/dinov2")

# Set BOP env vars
os.environ.setdefault("BOP_PATH", "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets")
os.environ.setdefault("BOP_OUTPUT_PATH", "D:/LDH_ws/foundpose-main/outputs")

if len(sys.argv) < 2:
    print("Usage: python run_pipeline.py <script> [args...]")
    sys.exit(1)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]

with open(script, encoding='utf-8') as f:
    code = f.read()

exec(compile(code, script, 'exec'), {'__name__': '__main__', '__file__': script})
