$ErrorActionPreference = "Stop"

$appRoot = Split-Path -Parent $PSCommandPath
$streamlitPath = Join-Path $appRoot ".venv\Scripts\streamlit.exe"
$stdoutLog = Join-Path $appRoot "streamlit.stdout.log"
$stderrLog = Join-Path $appRoot "streamlit.stderr.log"
$port = 8501
$localUrl = "http://localhost:$port"

if (-not (Test-Path $streamlitPath)) {
    throw "Streamlit launcher was not found: $streamlitPath"
}

$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $listener) {
    Start-Process -FilePath $streamlitPath `
        -ArgumentList @(
            "run",
            "app.py",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.port", "$port"
        ) `
        -WorkingDirectory $appRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog | Out-Null

    for ($i = 0; $i -lt 20; $i++) {
        $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($listener) {
            break
        }
        Start-Sleep -Milliseconds 200
    }
}

Start-Process $localUrl
