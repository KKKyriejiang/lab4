#!/bin/bash
set -e

# ---------- 编译 ----------
echo "[Compile]"
aarch64-linux-gnu-g++ -static -O3 -std=c++11 -lpthread -o gauss_neon_pthread gauss_neon_pthread.cpp


# ---------- 参数 ----------
SIZES=(64 128 256 384)      # 根据内存调整
THREADS=8                   # 改成 ARM 板子的核数
PARTS=("col" "row")         # 两种任务划分
ALG_NAMES=("serial" "dyn" "mutex" "sema" "bar")
CSV_TIME=results_time.csv
CSV_SPEED=results_speed.csv
mkdir -p results
echo "size,algo,part,time_ms"  > $CSV_TIME
echo "size,algo,part,speedup"  > $CSV_SPEED

# ---------- 采样 ----------
for N in "${SIZES[@]}"; do
  for part in "${PARTS[@]}"; do
    # 先跑串行
    base=$(./gauss_neon_pthread $N 1 0 $part)
    echo "$N,serial_$part,serial,$base" >> $CSV_TIME
    for algo in 1 2 3 4; do
      ms=$(./gauss_neon_pthread $N $THREADS $algo $part)
      echo "$N,${ALG_NAMES[$algo]}_$part,${ALG_NAMES[$algo]},$ms" >> $CSV_TIME
      sp=$(awk "BEGIN{print $base/$ms}")
      echo "$N,${ALG_NAMES[$algo]}_$part,${ALG_NAMES[$algo]},$sp" >> $CSV_SPEED
    done
  done
done
echo "✅ Raw data saved to $CSV_TIME / $CSV_SPEED"

# ---------- 作图 ----------
python3 - <<'PY'
import pandas as pd, matplotlib.pyplot as plt, seaborn, sys
import matplotlib; matplotlib.use("Agg")
time_df   = pd.read_csv("results_time.csv")
speed_df  = pd.read_csv("results_speed.csv")

# Execution-time plot (log-scale)
plt.figure(figsize=(8,5))
for key,grp in time_df.groupby("algo"):
    plt.plot(grp['size']**2, grp['time_ms']/1e3, 'o-', label=key)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Matrix Elements (N^2)'); plt.ylabel('Time (s, log)')
plt.title('ARM Gaussian Elimination — Execution Time')
plt.legend(); plt.tight_layout()
plt.savefig('results/time_plot.png', dpi=300)

# Speedup plot
plt.figure(figsize=(8,5))
for key,grp in speed_df.groupby("algo"):
    plt.plot(grp['size'], grp['speedup'], 'o-', label=key)
plt.xlabel('Matrix Size (N)'); plt.ylabel('Speedup'); plt.title('Speedup vs Serial')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig('results/speed_plot.png', dpi=300)
print("✅ Plots saved to results/")

