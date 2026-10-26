# ══════════════════════════════════════════════════════════════════════════════
#  QWEN 3.8 27B HERETIC (TRIAL 46) — KAGGLE BOOTSTRAP CELL
#  This is the ONLY cell you paste into the Kaggle notebook. It clones this repo
#  and runs the full production serving pipeline (server/serve.py).
#
#  One-time setup (30 seconds):
#    * Notebook options: Accelerator = 'GPU T4 x2' | Internet = ON
#    * Provide HF_TOKEN: either paste below, or Add-ons → Secrets → add 'HF_TOKEN'
#      (recommended: secrets survive notebook saves, the token stays private)
#
#  Every run gives you a NEW public URL + NEW API key — printed at the end as a
#  single line:  HERETIC_ENDPOINT={...}  → import with the client's `paste` cmd.
# ══════════════════════════════════════════════════════════════════════════════

REPO_URL = "https://github.com/Xero-cooks/Qwen-3-8-27b-abliterated-llama-cpp-server.git"
BRANCH   = "main"
HF_TOKEN = ""    # ← paste hf_... token here, or leave empty if using Kaggle Secret 'HF_TOKEN'

import os
import sys
import shutil
import subprocess

SRC = "/kaggle/tmp/serve-suite"

# Fresh clone every run → server fixes propagate by just re-running this cell.
if os.path.exists(SRC):
    shutil.rmtree(SRC)

r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, SRC])
if r.returncode != 0 or not os.path.exists(SRC + "/server/serve.py"):
    print("FATAL: repo clone failed — check REPO_URL, branch name, internet toggle.")
    raise SystemExit(1)

if HF_TOKEN:
    os.environ["HERETIC_HF_TOKEN"] = HF_TOKEN

print("🚀 serve.py launching in a clean subprocess — live [>>>>] stage markers below.")
print("   Stop the session anytime; nothing here needs manual cleanup.")
print()
subprocess.run([sys.executable, SRC + "/server/serve.py"])
