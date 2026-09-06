# Historical machine-specific run helper; not part of the FELT v2 pipeline.
$ErrorActionPreference = "Stop"

$CodeDir = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $CodeDir

$ReportPath = "..\01_data\02_output\logs\phase3_full_overwrite_workers5_report.csv"
$LogPath = "..\01_data\02_output\logs\phase3_full_overwrite_workers5.log"

Write-Host "Starting Phase 3 full raw tracking rerun."
Write-Host "Batch size: 1"
Write-Host "File-level workers: 5"
Write-Host "Py-Feat num_workers: 0"
Write-Host "Overwrite existing raw CSVs: yes"
Write-Host "Report: $ReportPath"
Write-Host "Log: $LogPath"
Write-Host ""

uv run python src/1_extract_raw_tracking.py `
  --batch-size 1 `
  --workers 5 `
  --pyfeat-num-workers 0 `
  --overwrite `
  --report $ReportPath `
  --log-file $LogPath

Write-Host ""
Write-Host "Phase 3 extraction command finished."
Write-Host "Report: $ReportPath"
Write-Host "Log: $LogPath"
