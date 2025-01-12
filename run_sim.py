import subprocess

def run_md(cwd, model: str, replicate: int):
    out = subprocess.check_output(["lmp", "-in", "../water.in", "-v", "model", model, "-v", "replicate", str(replicate)], cwd=cwd)
    time = None
    for oo in out.decode().split("\n"):
        if oo.startswith("Loop time of"):
            time = float(oo.split()[3])
    return time

with open("benchmark.out", "w", 1) as f:
    for model_type in ("dpa1l0", "dpa1l2", "dpa2"):
        if model_type == "dpa1l0":
            model_files = ["frozen_model.pb", "frozen_model_compressed.pb", "frozen_model.pth", "frozen_model_compressed.pth", "frozen_model.savedmodel"]
        elif model_type == "dpa1l2":
            model_files = ["frozen_model.pb", "frozen_model.pth", "frozen_model.savedmodel"]
        elif model_type == "dpa2":
            model_files = ["frozen_model.pth", "frozen_model.savedmodel"]
        else:
            raise ValueError("Unknown model type")
        for model_file in model_files:
            for ii in range(10):
                try:
                    time = run_md(model_type, model_file, 2**ii)
                except subprocess.CalledProcessError:
                    break
                print(model_type, model_file, ii, time, file=f)