#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS=""
    if command -v lscpu >/dev/null 2>&1; then
        OMP_NUM_THREADS=$(lscpu -p=core,socket 2>/dev/null |
                          grep -v '^#' | sort -u | wc -l | tr -d ' ')
        [[ "$OMP_NUM_THREADS" == "0" ]] && OMP_NUM_THREADS=""
    fi
    if [[ -z "$OMP_NUM_THREADS" ]] && command -v sysctl >/dev/null 2>&1; then
        OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu 2>/dev/null ||
                          sysctl -n hw.ncpu 2>/dev/null || echo "")
    fi
    if [[ -z "$OMP_NUM_THREADS" ]] && command -v nproc >/dev/null 2>&1; then
        OMP_NUM_THREADS=$(nproc 2>/dev/null || echo "")
    fi
    [[ -z "$OMP_NUM_THREADS" ]] && OMP_NUM_THREADS=1
fi
export OMP_NUM_THREADS
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-active}"

BIN="${BIN:-./build/cascade}"
SEED="${SEED:-12345}"
REALIZATIONS="${REALIZATIONS:-10}"
NPTS="${NPTS:-50}"
THETA_N="${THETA_N:-21}"
LOG_DIR="${LOG_DIR:-nohup}"
BACKGROUND=0
DRY_RUN=0

mkdir -p "$LOG_DIR"

usage() {
    cat <<'EOF'
Usage: ./run_tiers.sh [1|2|3|er|all] [options]

Options:
  --seed N            base seed (default 12345)
  --realizations N    realizations per run (default 10)
  --n-pts N           number of p values (default 50)
  --n-theta N         number of theta points (default 21)
  --dry-run           print commands without running them
  --background        launch each run with nohup in the background

Environment:
  OMP_NUM_THREADS     defaults to the physical-core count
  OMP_PROC_BIND       defaults to "spread"
  OMP_DYNAMIC         defaults to "false"
  OMP_WAIT_POLICY     defaults to "active"

Tier 3 repeats the L=100 scan as a regression check and is not part of "all".
EOF
}

TARGET="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        1|2|3|er|all)
            TARGET="$1"; shift ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --realizations)
            REALIZATIONS="$2"; shift 2 ;;
        --n-pts)
            NPTS="$2"; shift 2 ;;
        --n-theta)
            THETA_N="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --background)
            BACKGROUND=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ "$DRY_RUN" != "1" && ! -x "$BIN" ]]; then
    echo "Missing executable: $BIN" >&2
    echo "Build first: cmake -B build && cmake --build build -j" >&2
    exit 1
fi

run_case() {
    local name="$1"; shift
    local -a cmd=("$BIN" "$@" --seed "$SEED" --realizations "$REALIZATIONS" -p "$NPTS")
    local log="$LOG_DIR/$name.out"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s > %s 2>&1\n' "${cmd[*]}" "$log"
        return
    fi

    if [[ "$BACKGROUND" == "1" ]]; then
        nohup "${cmd[@]}" > "$log" 2>&1 &
    else
        "${cmd[@]}" > "$log" 2>&1
    fi
}

tier1() {
    echo "Tier 1: L=100, full theta grid, 10 zeta values"
    for z in 5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 45.0 50.0; do
        run_case "tier1-L100-z${z}" -L 100 -z "$z" \
            -n "$THETA_N" --tmin -1 --tmax 1
    done
}

tier2() {
    echo "Tier 2: L=20,40,60,80; 10 zeta/L ratios; full theta grid"
    echo "  L=100 is reused from Tier 1"
    for ratio in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
        for L in 20 40 60 80; do
            z=$(awk -v r="$ratio" -v l="$L" 'BEGIN { printf "%.10g", r * l }')
            run_case "tier2-L${L}-ratio${ratio}" -L "$L" -z "$z" \
                -n "$THETA_N" --tmin -1 --tmax 1
        done
    done
}

tier3() {
    echo "Tier 3: L=100 repeat of the Tier-1 scan (regression check)"
    for z in 5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 45.0 50.0; do
        run_case "tier3-L100-z${z}" -L 100 -z "$z" \
            -n "$THETA_N" --tmin -1 --tmax 1
    done
}

er_control() {
    echo "ER control: L=100, pure Erdos-Renyi graph"
    run_case "er-L100" -L 100 -z 0 -n "$THETA_N" --tmin -1 --tmax 1 \
        --topology er
}

case "$TARGET" in
    1) tier1 ;;
    2) tier2 ;;
    3) tier3 ;;
    er) er_control ;;
    all)
        tier1
        tier2
        er_control
        echo "Tier 3 is opt-in: ./run_tiers.sh 3"
        ;;
esac
