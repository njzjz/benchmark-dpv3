import json
import tempfile
import shutil
import subprocess
from pathlib import Path

def train(data, to_dir, tf, compress):
    to_dir = Path(to_dir)
    to_dir.mkdir(parents=True, exist_ok=True)
    data["training"]["numb_steps"] = 0
    data["training"]["training_data"]["systems"] = str(Path(__file__).parent / "data")
    data["training"]["validation_data"] = None
    returned = []
    with tempfile.TemporaryDirectory() as tmpdir:
        p_tmpdir = Path(tmpdir)
        with open(p_tmpdir / "input.json", "w") as f:
            json.dump(data, f)
        if tf:
            subprocess.run(["dp", "--tf", "train", "input.json"], cwd=str(p_tmpdir))
            subprocess.run(["dp", "--tf", "freeze"], cwd=str(p_tmpdir))
            returned.append(p_tmpdir / "frozen_model.pb")
            if compress:
                subprocess.run(["dp", "--tf", "compress"], cwd=str(p_tmpdir))
                returned.append(p_tmpdir / "frozen_model_compressed.pb")
        if True:  # pt
            subprocess.run(["dp", "--pt", "train", "input.json"], cwd=str(p_tmpdir))
            subprocess.run(["dp", "--pt", "freeze"], cwd=str(p_tmpdir))
            returned.append(p_tmpdir / "frozen_model.pth")
            if compress:
                subprocess.run(["dp", "--pt", "compress"], cwd=str(p_tmpdir))
                returned.append(p_tmpdir / "frozen_model_compressed.pth")
            subprocess.run(["dp", "convert-backend", "frozen_model.pth", "frozen_model.savedmodel"], cwd=str(p_tmpdir))
            returned.append(p_tmpdir / "frozen_model.savedmodel")
        for f in returned:
            if f.is_dir():
                # remove original directory
                if (to_dir / f.name).is_dir():
                    shutil.rmtree(to_dir / f.name)
                shutil.copytree(f, to_dir / f.name)
            elif f.is_file():
                shutil.copy(f, to_dir / f.name)
            else:
                raise RuntimeError("File not found")

with open("inputs/dpa1l0.json") as f:
    data = json.load(f)
train(data, "dpa1l0", tf=True, compress=True)
with open("inputs/dpa1l2.json") as f:
    data = json.load(f)
train(data, "dpa1l2", tf=True, compress=False)
with open("inputs/dpa2.json") as f:
    data = json.load(f)
train(data, "dpa2", tf=False, compress=False)
