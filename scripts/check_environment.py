#!/usr/bin/env python3
"""Check the active Python environment and optionally CUDA/model compatibility."""

# 1. 경로 및 설정
import argparse
import datetime as dt
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 이 스크립트에서 사용하는 로그 보조 함수
def timestamp(): return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
def now_local(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")
class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True); self.handle = path.open("a", encoding="utf-8", buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
    def start(self): sys.stdout = sys.stderr = self
    def write(self, text): self.stdout.write(text); self.stdout.flush(); self.handle.write(text); self.handle.flush(); return len(text)
    def flush(self): self.stdout.flush(); self.handle.flush()
    def close(self): sys.stdout, sys.stderr = self.stdout, self.stderr; self.handle.close()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-check", action="store_true")
    args = parser.parse_args()
    log_path = PROJECT_ROOT / "logs/preprocess" / f"environment_{timestamp()}.log"
    tee = Tee(log_path)
    tee.start()
    report = {"python": sys.version, "executable": sys.executable, "started": now_local(), "packages": {}, "errors": []}
    try:
        for package in ["torch", "transformers", "huggingface_hub", "datasets", "accelerate", "tokenizers",
                        "safetensors", "numpy", "pandas", "pyarrow", "sklearn", "matplotlib", "yaml", "psutil"]:
            try:
                module = importlib.import_module(package)
                report["packages"][package] = getattr(module, "__version__", "unknown")
                print(f"[OK] {package:18s} {report['packages'][package]}")
            except Exception as exc:
                report["errors"].append(f"{package}: {exc}")
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        report["pip_check"] = check.stdout.strip()
        if check.returncode:
            report["errors"].append("pip check failed")
        # 전처리에 필요한 외부 실행 파일도 함께 확인한다.
        for tool, version_args in {"mmseqs": ["version"], "skani": ["--version"]}.items():
            executable = shutil.which(tool)
            report[tool] = executable
            if executable:
                version = subprocess.run(
                    [executable, *version_args], capture_output=True, text=True, check=True
                )
                report[f"{tool}_version"] = (version.stdout or version.stderr).strip()
            else:
                report["errors"].append(f"{tool} executable missing from PATH")
        if args.gpu_check:
            import torch
            report["cuda_version"] = torch.version.cuda
            report["gpu_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if not torch.cuda.is_available():
                report["errors"].append("CUDA device not accessible")
            else:
                a = torch.ones((128, 128), device="cuda")
                report["gpu_matmul_result"] = float((a @ a)[0,0])
                assert report["gpu_matmul_result"] == 128.0
        report["completed"] = now_local()
        report["passed"] = not report["errors"]
        write_json(PROJECT_ROOT / "logs/preprocess/environment_latest.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
