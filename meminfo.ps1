$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$usedPct = [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100, 1)
Write-Output "Total RAM: ${totalGB} GB"
Write-Output "Free RAM:  ${freeGB} GB"
Write-Output "Used %:    ${usedPct}%"
Write-Output ""
Write-Output "Python processes:"
Get-Process python -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | ForEach-Object {
    $mb = [math]::Round($_.WorkingSet64 / 1MB)
    Write-Output "  PID $($_.Id): ${mb} MB ($($_.PriorityClass))"
}
Write-Output ""
Write-Output "Top 5 processes by memory:"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 8 | ForEach-Object {
    $mb = [math]::Round($_.WorkingSet64 / 1MB)
    Write-Output "  $($_.ProcessName) PID=$($_.Id): ${mb} MB"
}
