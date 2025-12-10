#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent
candidates = [
    BASE / "gui" / "enhanced_app.py",   # Pro GUI
    BASE / "gui" / "main_app.py",       # Legacy GUI-in-folder
    BASE / "enhanced_app.py",           # Pro GUI at root
    BASE / "main_app.py",               # Legacy GUI at root
]

# Ensure the project root is importable even if GUI lives in gui/
env = os.environ.copy()
env["PYTHONPATH"] = str(BASE) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

for target in candidates:
    if target.exists():
        print(f"▶ Launching: {target}")
        sys.exit(subprocess.call([sys.executable, str(target)], env=env))

print("❌ No GUI entry found. Looked for:", *map(str, candidates), sep="\n  - ")
sys.exit(1)
