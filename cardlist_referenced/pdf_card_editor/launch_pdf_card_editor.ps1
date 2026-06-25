$ErrorActionPreference = "Stop"

$appRoot = Split-Path -Parent $PSCommandPath
$streamlitPath = Join-Path $appRoot ".venv\Scripts\streamlit.exe"
$stdoutLog = Join-Path $appRoot "streamlit.stdout.log"
$stderrLog = Join-Path $appRoot "streamlit.stderr.log"
$port = 8501
$localUrl = "http://localhost:$port"
$startupTimeoutSeconds = 30
$pollIntervalMilliseconds = 500

function Show-LauncherMessage {
    param(
        [string]$message,
        [string]$title = "Card PDF Filter Launcher",
        [int]$popupType = 16
    )

    try {
        $shell = New-Object -ComObject WScript.Shell
        $null = $shell.Popup($message, 0, $title, $popupType)
    }
    catch {
        Write-Host "$title`n$message"
    }
}

function Get-LogTail {
    param(
        [string]$path,
        [int]$lineCount = 20
    )

    if (-not (Test-Path $path)) {
        return "(log file not found)"
    }

    $lines = Get-Content $path -Tail $lineCount -ErrorAction SilentlyContinue
    if (-not $lines) {
        return "(log file is empty)"
    }

    return ($lines -join [Environment]::NewLine)
}

try {
    if (-not (Test-Path $streamlitPath)) {
        throw "Streamlit launcher was not found: $streamlitPath"
    }

    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) {
        Set-Content -Path $stdoutLog -Value "" -Encoding UTF8
        Set-Content -Path $stderrLog -Value "" -Encoding UTF8

        $streamlitProcess = Start-Process -FilePath $streamlitPath `
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
            -RedirectStandardError $stderrLog `
            -PassThru

        $deadline = (Get-Date).AddSeconds($startupTimeoutSeconds)
        do {
            if ($streamlitProcess.HasExited) {
                $stderrTail = Get-LogTail -path $stderrLog
                throw "Streamlit exited before startup completed. ExitCode=$($streamlitProcess.ExitCode)`n`nLast stderr lines:`n$stderrTail"
            }

            Start-Sleep -Milliseconds $pollIntervalMilliseconds
            $streamlitProcess.Refresh()
            $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        } while (-not $listener -and (Get-Date) -lt $deadline)

        if (-not $listener) {
            $stdoutTail = Get-LogTail -path $stdoutLog
            $stderrTail = Get-LogTail -path $stderrLog
            throw "Timed out waiting for Streamlit to listen on port $port within $startupTimeoutSeconds seconds.`n`nLast stdout lines:`n$stdoutTail`n`nLast stderr lines:`n$stderrTail"
        }
    }

    Start-Process $localUrl | Out-Null
}
catch {
    $message = @(
        "PDF card editor failed to start."
        ""
        $_.Exception.Message
        ""
        "Logs:"
        "stdout: $stdoutLog"
        "stderr: $stderrLog"
        ""
        "If needed, try launch_pdf_card_editor.cmd to see the startup behavior more directly."
    ) -join [Environment]::NewLine

    Show-LauncherMessage -message $message
    exit 1
}
