$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$leanCmd = Get-Command lean -ErrorAction SilentlyContinue
if ($null -eq $leanCmd -and $env:LEAN_EXE) {
    if (Test-Path -LiteralPath $env:LEAN_EXE) { $leanCmd = Get-Item -LiteralPath $env:LEAN_EXE }
}
if ($null -eq $leanCmd) { throw 'Lean executable not found.' }

$z3Cmd = Get-Command z3 -ErrorAction SilentlyContinue
if ($null -eq $z3Cmd -and $env:Z3_EXE) {
    if (Test-Path -LiteralPath $env:Z3_EXE) { $z3Cmd = Get-Item -LiteralPath $env:Z3_EXE }
}
if ($null -eq $z3Cmd) { throw 'Z3 executable not found.' }

$leanPath = if ($leanCmd.PSObject.Properties.Name -contains 'Source' -and $leanCmd.Source) { $leanCmd.Source } else { $leanCmd.FullName }
$z3Path = if ($z3Cmd.PSObject.Properties.Name -contains 'Source' -and $z3Cmd.Source) { $z3Cmd.Source } else { $z3Cmd.FullName }

$leanFile = Join-Path $PSScriptRoot 'LeanChecks.lean'
$z3File = Join-Path $PSScriptRoot 'z3\invariants.smt2'

& $leanPath $leanFile
if ($LASTEXITCODE -ne 0) { throw "Lean failed with exit code $LASTEXITCODE." }

$z3Output = @(& $z3Path $z3File)
if ($LASTEXITCODE -ne 0) { throw "Z3 failed with exit code $LASTEXITCODE." }
$unsatCount = @($z3Output | Where-Object { $_.Trim() -eq 'unsat' }).Count
if ($unsatCount -ne 3) {
    $joined = $z3Output -join '; '
    throw "Expected three unsat results from Z3, got ${unsatCount}: $joined"
}

Write-Output 'LEAN: PASS'
Write-Output 'Z3: PASS (3/3 unsat checks)'
