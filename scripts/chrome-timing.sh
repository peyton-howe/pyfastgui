#!/usr/bin/env sh
# Run the chrome timing benchmark and save the results with this machine's CPU model, so
# numbers from different devices can be compared. CPU-only: needs a Rust toolchain, but no
# GPU, display, Vulkan SDK or Python.
#
#   scripts/chrome-timing.sh            # writes chrome-timing-<host>-<date>.txt in the repo root
#
# Windows: scripts/chrome-timing.ps1
set -eu

cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found - install Rust from https://rustup.rs first" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)" ;;
    Linux)  cpu="$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)"
            # ARM boards often have no "model name"; lscpu knows the core type.
            [ -n "$cpu" ] || cpu="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
            [ -n "$cpu" ] || cpu=unknown ;;
    *)      cpu=unknown ;;
esac

host="$(hostname 2>/dev/null | cut -d. -f1)"
out="chrome-timing-${host:-host}-$(date +%Y%m%d-%H%M%S).txt"

echo "Building (release)..."
cargo build --release -q -p fastgui-chrome --example chrome_bench

{
    echo "host: ${host:-unknown}"
    echo "cpu:  $cpu"
    echo "os:   $(uname -srm)"
    echo "rust: $(rustc --version)"
    echo
    cargo run --release -q -p fastgui-chrome --example chrome_bench
} | tee "$out"

echo
echo "Saved to $out"
