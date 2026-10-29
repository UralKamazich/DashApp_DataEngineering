param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-Native {
    param([string]$Description, [scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Ensure-Electron {
    # Newer npm releases skip dependency install scripts unless approved, and
    # Electron 28's own downloader can leave a partial dist on recent Node.js.
    # Verify the binary and install it from the official release if needed.
    $ElectronDir = Join-Path $Root "node_modules\electron"
    $ElectronExe = Join-Path $ElectronDir "dist\electron.exe"
    $PathFile = Join-Path $ElectronDir "path.txt"
    if ((Test-Path $ElectronExe) -and (Test-Path $PathFile)) {
        return
    }

    $Version = (Get-Content (Join-Path $ElectronDir "package.json") -Raw | ConvertFrom-Json).version
    $Arch = (& node -p "process.arch").Trim()
    $Name = "electron-v$Version-win32-$Arch.zip"
    $CacheDir = Join-Path $env:LOCALAPPDATA "electron\Cache"
    $Zip = Join-Path $CacheDir $Name
    New-Item -ItemType Directory -Force $CacheDir | Out-Null
    if (-not (Test-Path $Zip)) {
        Write-Host "Downloading Electron $Version ($Arch)..."
        $Partial = "$Zip.partial"
        Invoke-WebRequest "https://github.com/electron/electron/releases/download/v$Version/$Name" -OutFile $Partial
        Move-Item -Force $Partial $Zip
    }

    $Dist = Join-Path $ElectronDir "dist"
    Remove-Item -Recurse -Force $Dist -ErrorAction SilentlyContinue
    Expand-Archive $Zip $Dist
    [IO.File]::WriteAllText($PathFile, "electron.exe")
    if (-not (Test-Path $ElectronExe)) {
        throw "Electron binary is missing after installation: $ElectronExe"
    }
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating Windows Python 3.14 environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Native "venv" { py -3.14 -m venv .venv }
    } else {
        Invoke-Native "venv" { python -m venv .venv }
    }
}

if (-not $SkipInstall) {
    Invoke-Native "pip upgrade" { & $Python -m pip install --upgrade pip }
    Invoke-Native "pip install" { & $Python -m pip install -r requirements.txt }
    if (-not (Test-Path (Join-Path $Root "node_modules"))) {
        Invoke-Native "npm ci" { npm ci }
    }
}
Ensure-Electron

# Some terminal hosts define this for their own Electron integration. It must
# not leak into DataAnalize, otherwise Electron starts as a Node.js process.
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
npm run dev
