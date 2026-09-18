"""从 E:\\BAAI\\910b 下 FlagOpBench 的 *_ascend.json 对生成国产昇腾表。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(r"E:\BAAI\910b")
SCRIPTS = Path(r"E:\FlagOpBench\scripts")
OUT_XLSX = ROOT / "国产_ascend_910B_FlagOpBench测试结果.xlsx"


def main() -> int:
    pairs = []
    for base in ROOT.rglob("*_ascend.json"):
        name = base.name
        if name.endswith("_flagos_ascend.json") or "_compare_" in name:
            continue
        if not name.endswith("_ascend.json"):
            continue
        stem = name[: -len("_ascend.json")]
        flagos = base.with_name(f"{stem}_flagos_ascend.json")
        if flagos.exists():
            pairs.append((base, flagos, stem))

    if not pairs:
        print(f"No *_ascend.json pairs under {ROOT}")
        print("请把机上 results/<op>/*_ascend.json 与 *_flagos_ascend.json 拷到例如:")
        print(rf"  {ROOT}\swiglu\")
        return 1

    cmp_dir = ROOT / "_compare_tmp"
    cmp_dir.mkdir(exist_ok=True)
    print(f"pairs={len(pairs)}")
    for base, flagos, stem in sorted(pairs, key=lambda x: x[2]):
        out = cmp_dir / f"{stem}_compare_ascend.json"
        subprocess.check_call(
            [
                sys.executable,
                str(SCRIPTS / "gen_compare_result.py"),
                "--baseline",
                str(base),
                "--flagos",
                str(flagos),
                "--output",
                str(out),
            ]
        )
        # per-op xlsx
        op_xlsx = base.parent / f"冒烟_ascend_{stem}_FlagOpBench.xlsx"
        subprocess.check_call(
            [
                sys.executable,
                str(SCRIPTS / "gen_nv_style_xlsx.py"),
                "--compare",
                str(out),
                "--output",
                str(op_xlsx),
            ]
        )

    subprocess.check_call(
        [
            sys.executable,
            str(SCRIPTS / "gen_nv_style_xlsx.py"),
            "--compare-dir",
            str(cmp_dir),
            "--output",
            str(OUT_XLSX),
        ]
    )
    print(f"master: {OUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
