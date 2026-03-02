import re

with open("inference.py", "r") as f:
    code = f.read()

# remove inline import of scipy.stats
code = code.replace("    import scipy.stats as stats\n", "")

# add it to the top
if "import scipy.stats as stats" not in code[:500]:
    code = code.replace("import torch", "import torch\nimport scipy.stats as stats\nimport pandas as pd", 1)

with open("inference.py", "w") as f:
    f.write(code)
