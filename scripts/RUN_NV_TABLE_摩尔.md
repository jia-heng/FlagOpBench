# 摩尔机 NV 同款表 — JumpServer 一键命令

本机 zip：`E:\FlagOpBench-nv-table.zip`（解压根目录 `FlagOpBench-new/`）

## 1. 宿主机 `10.121.39.1`：上传并拷进容器

JumpServer 文件管理把 zip 传到例如 `/data/yangyifei/` 后：

```bash
mkdir -p /data/yangyifei/works
# 若 zip 在家目录下载处，先 mv 到 /data/yangyifei/
ls -lh /data/yangyifei/FlagOpBench-nv-table.zip || ls -lh ~/FlagOpBench-nv-table.zip

docker start yangyifei-vllm-musa 2>/dev/null || true
docker cp /data/yangyifei/FlagOpBench-nv-table.zip yangyifei-vllm-musa:/workspace/works/
docker exec -it yangyifei-vllm-musa bash
```

## 2. 容器内：解压 + 门禁 + 跑表

```bash
export MUSA_VISIBLE_DEVICES=0,1,2,3
export GEMS_VENDOR=mthreads
cd /workspace/works
python - <<'PY'
import zipfile
from pathlib import Path
z = Path("FlagOpBench-nv-table.zip")
with zipfile.ZipFile(z) as zf:
    zf.extractall(".")
print("ok", Path("FlagOpBench-new").exists())
PY

cd /workspace/works/FlagOpBench-new

# 门禁
python -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.get_device_name(0))"
python -c "import flag_gems; print('flag_gems OK', flag_gems.__file__)"

# 若 flag_gems 失败：先查是否已有仓，再按 FlagGems setup（需外网或离线包）
# ls /workspace /data 2>/dev/null | head
# find /workspace /data -maxdepth 3 -iname 'FlagGems' 2>/dev/null

# 只跑 swiglu 出表
bash scripts/run_mthreads_nv_table.sh

# 通过后扩 pilot4
# RUN_PILOT4=1 bash scripts/run_mthreads_nv_table.sh
```

冒烟（更快）：

```bash
WARMUP=2 REPEAT=5 bash scripts/run_mthreads_nv_table.sh
```

## 3. 拉回结果

容器内：

```bash
ls -lh results/国产_mthreads_测试结果.xlsx results/swiglu/
```

宿主机：

```bash
docker cp yangyifei-vllm-musa:/workspace/works/FlagOpBench-new/results /data/yangyifei/mthreads_nv_results
ls -lh /data/yangyifei/mthreads_nv_results
```

把 `国产_mthreads_测试结果.xlsx` 和 `swiglu/*_compare_*.json` 下载到本机即可。
