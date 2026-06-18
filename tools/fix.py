import re

from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
target = repo_root / "src" / "ma_vlcm" / "inference.py"

with open(target, "r") as f:
    code = f.read()

# remove inline import of scipy.stats
code = code.replace("    import scipy.stats as stats\n", "")

# add it to the top
if "import scipy.stats as stats" not in code[:500]:
    code = code.replace("import torch", "import torch\nimport scipy.stats as stats\nimport pandas as pd", 1)

with open(target, "w") as f:
    f.write(code)
