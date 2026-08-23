$procs = Get-Process python -ErrorAction SilentlyContinue
foreach ($p in $procs) {
    try {
        $p.PriorityClass = 'BelowNormal'
        $mem = [math]::Round($p.WorkingSet64/1MB)
        Write-Output "PID=$($p.Id) priority=$($p.PriorityClass) mem=${mem}MB"
    } catch {
        Write-Output "PID=$($p.Id) failed: $_"
    }
}
