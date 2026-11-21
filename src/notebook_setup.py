import os
from pathlib import Path

def set_repo_root():
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)