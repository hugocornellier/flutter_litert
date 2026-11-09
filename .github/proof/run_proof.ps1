# TEMPORARY (feat/4.1.0-desktop only, remove before merge).
# Runs the release-built proof app in one mode and prints its log.
# Usage: pwsh -File run_proof.ps1 <interpreter|compiled> <tag> <release dir> <out dir>
# Exits with the app's exit code (99 on timeout, 98 when no log was written).
param(
  [Parameter(Mandatory)] [string] $Mode,
  [Parameter(Mandatory)] [string] $Tag,
  [Parameter(Mandatory)] [string] $ReleaseDir,
  [Parameter(Mandatory)] [string] $OutDir
)
$ReleaseDir = (Resolve-Path $ReleaseDir).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$out = Join-Path (Resolve-Path $OutDir).Path "proof-$Tag-$Mode.txt"
if (Test-Path $out) { Remove-Item $out }
$exe = Join-Path $ReleaseDir 'flutter_litert_example.exe'
Write-Host "== ${Tag} / ${Mode}: $exe (PATH=$env:PATH)"
$p = Start-Process -FilePath $exe -ArgumentList $Mode, $out -WorkingDirectory $ReleaseDir -PassThru
$null = $p.Handle
if (-not $p.WaitForExit(300000)) {
  $p.Kill()
  Write-Host "TIMEOUT after 300 s"
  exit 99
}
$code = $p.ExitCode
if (Test-Path $out) {
  Get-Content $out | ForEach-Object { Write-Host "  | $_" }
} else {
  Write-Host "no log written (exit code $code)"
  "no log written (exit code $code)" | Set-Content $out
  if ($code -eq 0) { $code = 98 }
}
Write-Host "== ${Tag} / ${Mode} exit code: $code"
exit $code
