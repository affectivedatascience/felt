param(
    [int]$ExpectedCsvCount = 2452,
    [int]$IntervalSeconds = 300
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$outputRoot = Join-Path $projectRoot "01_data\02_output"
$rawRoot = Join-Path $outputRoot "01_raw_motion"
$logRoot = Join-Path $outputRoot "logs"
$historyPath = Join-Path $logRoot "pyfeat_v2_full_progress.csv"
$latestPath = Join-Path $logRoot "pyfeat_v2_full_progress_latest.json"
$shardReports = 1..3 | ForEach-Object {
    Join-Path $logRoot "pyfeat_v2_full_shard${_}_report.csv"
}
$consoleLogs = 1..3 | ForEach-Object {
    Join-Path $logRoot "pyfeat_v2_full_shard${_}_console.log"
}

if (-not (Test-Path $historyPath)) {
    "timestamp,csv_count,expected_count,percent,actors_with_output,completed_shards,logged_errors,gpu_utilization_percent,gpu_memory_mib,status" |
        Set-Content -LiteralPath $historyPath -Encoding utf8
}

while ($true) {
    $csvFiles = @(Get-ChildItem -LiteralPath $rawRoot -Recurse -File -Filter *.csv -ErrorAction SilentlyContinue)
    $csvCount = $csvFiles.Count
    $actorCount = @($csvFiles | Group-Object { $_.Directory.Name }).Count
    $completedShards = 0
    foreach ($reportPath in $shardReports) {
        if (-not (Test-Path -LiteralPath $reportPath)) {
            continue
        }
        $reportRows = @(Import-Csv -LiteralPath $reportPath)
        if ($reportRows.Count -gt 0) {
            $taskCount = [int]$reportRows[0].task_count
            if ($reportRows.Count -eq $taskCount) {
                $completedShards++
            }
        }
    }
    $loggedErrors = @(
        Select-String -Path $consoleLogs -Pattern "Error processing file|Traceback" -ErrorAction SilentlyContinue
    ).Count

    $gpuUtilization = ""
    $gpuMemory = ""
    try {
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null) -split ","
        if ($gpu.Count -ge 2) {
            $gpuUtilization = $gpu[0].Trim()
            $gpuMemory = $gpu[1].Trim()
        }
    } catch {
        # GPU telemetry is optional; extraction status remains available.
    }

    $status = if ($csvCount -ge $ExpectedCsvCount) {
        "complete"
    } elseif ($completedShards -eq $shardReports.Count) {
        "shards_complete_with_shortfall"
    } else {
        "running"
    }

    $snapshot = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        csv_count = $csvCount
        expected_count = $ExpectedCsvCount
        percent = [math]::Round(100 * $csvCount / $ExpectedCsvCount, 2)
        actors_with_output = $actorCount
        completed_shards = $completedShards
        logged_errors = $loggedErrors
        gpu_utilization_percent = $gpuUtilization
        gpu_memory_mib = $gpuMemory
        status = $status
    }

    ($snapshot.Values -join ",") | Add-Content -LiteralPath $historyPath -Encoding utf8
    $snapshot | ConvertTo-Json | Set-Content -LiteralPath $latestPath -Encoding utf8
    Write-Output ($snapshot | ConvertTo-Json -Compress)

    if ($status -ne "running") {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
