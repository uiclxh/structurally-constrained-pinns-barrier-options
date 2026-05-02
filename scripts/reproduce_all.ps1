param(
    [switch]$SkipHeavy,
    [switch]$StopOnError = $true
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Invoke-Chapter {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [string]$ScriptPath
    )

    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host "Running: python $ScriptPath"

    python $ScriptPath
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $message = "$Label failed with exit code $exitCode."
        if ($StopOnError) {
            throw $message
        }
        Write-Warning $message
    }
}

Write-Host "Repository root: $RepoRoot"
Write-Host "Installing dependencies is not performed automatically."
Write-Host "Run first if needed: python -m pip install -r requirements.txt"

Invoke-Chapter "Chapter 3: FDM benchmark" "src/chapter3_fdm_benchmark_only.py"
Invoke-Chapter "Chapter 4: Barrier surrogate framework" "src/chapter4_barrier_surrogate_framework.py"
Invoke-Chapter "Chapter 5: Validation protocol" "src/chapter5_validation_protocol_framework.py"
Invoke-Chapter "Chapter 6: Experimental design" "src/chapter6_experimental_design_framework.py"

if ($SkipHeavy) {
    Write-Host ""
    Write-Host "Skipping heavy chapters because -SkipHeavy was provided." -ForegroundColor Yellow
    Write-Host "Skipped: Chapter 7 formal ablation, Chapter 8 accuracy workflow, Chapter 9 runtime workflow."
} else {
    Invoke-Chapter "Chapter 7: Formal ablation and failure diagnostics" "src/chapter7_ablation_failure_diagnostics_real.py"
    Invoke-Chapter "Chapter 8: Accuracy, Greeks, boundary consistency, and residual diagnostics" "src/chapter8_results_accuracy_real.py"
    Invoke-Chapter "Chapter 9: Runtime and deployment economics" "src/chapter9_results_runtime_real.py"
}

Invoke-Chapter "Chapter 10: Discussion and roadmap" "src/chapter10_discussion_roadmap_framework.py"

Write-Host ""
Write-Host "Workflow complete." -ForegroundColor Green
Write-Host "Review regenerated chapter output folders before replacing curated files under results/, figures/, tables/, or models/."
