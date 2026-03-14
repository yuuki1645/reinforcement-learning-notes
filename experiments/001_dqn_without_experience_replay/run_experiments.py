# type: ignore

"""
実験ランナー: main.py を results/run001.json ～ run010.json で順に実行する。

使い方:
  python run_experiments.py

このスクリプトがあるディレクトリをカレントにして実行すること。
"""

import subprocess
import sys
from pathlib import Path

# このスクリプトがあるディレクトリ（main.py と results/ の基準）
SCRIPT_DIR = Path(__file__).resolve().parent
NUM_RUNS = 10


def main():
    for i in range(1, NUM_RUNS + 1):
        output_path = SCRIPT_DIR / "results" / f"run{i:03d}.json"
        print(f"\n--- Run {i}/{NUM_RUNS}: {output_path.name} ---")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "main.py"), str(output_path)],
            cwd=SCRIPT_DIR,
        )
        if result.returncode != 0:
            print(f"Run {i} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    print(f"\n全 {NUM_RUNS} 回の実験が完了しました。")


if __name__ == "__main__":
    main()
