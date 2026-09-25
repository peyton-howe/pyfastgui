# Run the chrome timing benchmark and save the results with this machine's CPU model, so
# numbers from different devices can be compared. CPU-only: needs a Rust toolchain, but no
# GPU, Vulkan SDK or Python.
#
#   powershell -ExecutionPolicy Bypass -File scripts\chrome-timing.ps1
#
# Writes chrome-timing-<host>-<date>.txt in the repo root. macOS/Linux: scripts/chrome-timing.sh
$ErrorActionPreference = 'Stop'
# cargo prints UTF-8 (× in the table); decode it as such when capturing.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location (Join-Path $PSScriptRoot '..')

# rustup installs here, but it isn't always on PATH in a fresh shell (see ROADMAP "Environment").
$cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (-not (Get-Command cargo -ErrorAction SilentlyContinue) -and (Test-Path $cargoBin)) {
    $env:Path = "$cargoBin;$env:Path"
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error 'cargo not found - install Rust from https://rustup.rs first'
}

$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$os = (Get-CimInstance Win32_OperatingSystem).Caption
$hostName = $env:COMPUTERNAME
$out = "chrome-timing-$hostName-$(Get-Date -Format yyyyMMdd-HHmmss).txt"

Write-Host 'Building (release)...'
cargo build --release -q -p fastgui-chrome --example chrome_bench
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$header = @(
    "host: $hostName"
    "cpu:  $cpu"
    "os:   $os"
    "rust: $(rustc --version)"
    ''
)
$bench = cargo run --release -q -p fastgui-chrome --example chrome_bench
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$lines = $header + $bench
$lines | ForEach-Object { Write-Host $_ }
# UTF-8 so the × in the table survives.
$lines | Set-Content -Encoding utf8 $out

Write-Host ''
Write-Host "Saved to $out"
