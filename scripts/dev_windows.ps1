param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating Windows Python 3.14 environment..."
    py -3.14 -m venv .venv
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
    if (-not (Test-Path (Join-Path $Root "node_modules"))) {
        npm ci
    }
}

# Some terminal hosts define this for their own Electron integration. It must
# not leak into DataAnalize, otherwise Electron starts as a Node.js process.
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
npm run dev
