# PowerShell Script to Prepare GitHub Portfolio
# Destination: C:\Users\macht\Scar

$DestRoot = ".\SGR_Core_Release"
$SourceSA_C = ".."
$SourceSA_D = "D:\SA"

Write-Host "Starting Portfolio Staging to $DestRoot..." -ForegroundColor Cyan

# 1. Create Root Directory
if (!(Test-Path $DestRoot)) {
    New-Item -ItemType Directory -Path $DestRoot | Out-Null
    Write-Host "Created $DestRoot" -ForegroundColor Green
}

# Helper Function for Robocopy
function Copy-Project {
    param (
        [string]$SourcePath,
        [string]$DestName,
        [string[]]$ExcludesDir,
        [string[]]$ExcludesFile
    )

    $DestPath = "$DestRoot\$DestName"
    Write-Host "`nProcessing $DestName..." -ForegroundColor Yellow
    
    # Common Exclusions
    $GlobalExcludeDirs = @(".git", ".idea", ".vscode", "venv", "__pycache__", "build", "dist", ".gradle", "captures")
    $GlobalExcludeFiles = @(".env", ".DS_Store", "*.log", "*.pyc", "local.properties", "*.hprof", "memory.db", "sgr_core.log")

    $Dirs = $GlobalExcludeDirs + $ExcludesDir
    $Files = $GlobalExcludeFiles + $ExcludesFile

    # Construct Arguments
    $cmdArgs = @($SourcePath, $DestPath, "/E", "/XD")
    $cmdArgs += $Dirs
    $cmdArgs += @("/XF")
    $cmdArgs += $Files
    $cmdArgs += @("/NFL", "/NDL", "/NJH", "/NJS")
    
    # Execute Robocopy
    & robocopy $cmdArgs
    
    # Check Exit Code (0-7 is success)
    if ($LASTEXITCODE -ge 8) {
        Write-Host "Error copying $DestName (Code: $LASTEXITCODE)" -ForegroundColor Red
    } else {
        Write-Host "Copied $DestName" -ForegroundColor Green
    }
    
    # Rename README (English)
    $GithubReadme = "$DestPath\GITHUB_README.md"
    $Readme = "$DestPath\README.md"
    
    if (Test-Path $GithubReadme) {
        if (Test-Path $Readme) { Remove-Item $Readme -Force }
        Rename-Item $GithubReadme "README.md"
        Write-Host "Renamed GITHUB_README.md to README.md" -ForegroundColor White
    }

    # Rename README (Russian)
    $GithubReadmeRU = "$DestPath\GITHUB_README_RU.md"
    $ReadmeRU = "$DestPath\README_RU.md"

    if (Test-Path $GithubReadmeRU) {
        if (Test-Path $ReadmeRU) { Remove-Item $ReadmeRU -Force }
        Rename-Item $GithubReadmeRU "README_RU.md"
        Write-Host "Renamed GITHUB_README_RU.md to README_RU.md" -ForegroundColor White
    }
}


# --- PROCESS PROJECTS ---

# 1. sgr_core (From C:)
Copy-Project -SourcePath "$SourceSA_C\sgr_core" -DestName "sgr_core" -ExcludesDir @("qdrant_data", "qdrant_data_indexer", "logs", "proxy_data", "generated_files") -ExcludesFile @()

# 2. bcs_manager (From D:)
Copy-Project -SourcePath "$SourceSA_D\bcs_manager" -DestName "bcs_manager" -ExcludesDir @("build", "app\build") -ExcludesFile @("local.properties")

# 3. personal_assistant (From D:)
Copy-Project -SourcePath "$SourceSA_D\personal_assistant" -DestName "personal_assistant" -ExcludesDir @() -ExcludesFile @(".env")

# 4. cbr_rag (From D:)
Copy-Project -SourcePath "$SourceSA_D\cbr_rag" -DestName "cbr_rag" -ExcludesDir @() -ExcludesFile @()

Write-Host "`nPortfolio Staging Complete!" -ForegroundColor Cyan
Write-Host "Files are ready in: $DestRoot" -ForegroundColor Cyan
Write-Host "Next Step: cd $DestRoot; git init" -ForegroundColor Yellow
