param(
    [string]$ContainerName = "openclaw-sglang-local",
    [string]$Image = "lmsysorg/sglang:latest-runtime",
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
    [int]$Port = 30000,
    [string]$MemFractionStatic = "0.35",
    [string]$ShmSize = "8g",
    [int]$ReadyChecks = 120
)

$ErrorActionPreference = "Stop"

$modelsUrl = "http://127.0.0.1:$Port/v1/models"
try {
    $response = Invoke-WebRequest -Uri $modelsUrl -UseBasicParsing -TimeoutSec 3
    if ($response.StatusCode -eq 200) {
        Write-Host "SGLang is already ready."
        $response.Content
        exit 0
    }
}
catch {
}

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
$docker = Join-Path $dockerBin "docker.exe"
if (-not (Test-Path $docker)) {
    throw "Docker CLI not found at $docker. Install/start Docker Desktop first."
}
$env:PATH = "$dockerBin;$env:PATH"

function Invoke-Docker {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$DockerArgs)
    & $docker @DockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
New-Item -ItemType Directory -Force -Path $hfCache | Out-Null

$existing = & $docker ps -a --filter "name=^/$ContainerName$" --format "{{.Names}} {{.Status}}"
if ($LASTEXITCODE -ne 0) {
    throw "Cannot access Docker Desktop. Start Docker Desktop and make sure this user can access the Docker engine."
}
if ($existing) {
    $running = & $docker inspect -f "{{.State.Running}}" $ContainerName
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot inspect container: $ContainerName"
    }
    if ($running -eq "true") {
        Write-Host "Container is already running: $ContainerName"
    }
    else {
        Write-Host "Removing stopped container: $ContainerName"
        Invoke-Docker rm $ContainerName | Out-Null
    }
}

if (-not (& $docker ps --filter "name=^/$ContainerName$" --format "{{.Names}}")) {
    $localImage = & $docker image inspect $Image --format "{{.Id}}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $localImage) {
        Write-Host "Pulling image if needed: $Image"
        Invoke-Docker pull $Image
    }
    else {
        Write-Host "Using local image: $Image"
    }

    Write-Host "Starting SGLang container: $ContainerName"
    $runArgs = @(
        "run", "-d",
        "--name", $ContainerName,
        "--gpus", "all",
        "--shm-size", $ShmSize,
        "-e", "SGLANG_PREFIX_CACHE_DEBUG_LOG=1",
        "-p", "${Port}:30000",
        "-v", "${hfCache}:/root/.cache/huggingface",
        $Image,
        "bash", "-lc",
        "python3 -m pip install --no-cache-dir distro && python3 -m sglang.launch_server --model-path $Model --host 0.0.0.0 --port 30000 --enable-metrics --log-requests --log-requests-level 1 --log-requests-format json --radix-eviction-policy lru --mem-fraction-static $MemFractionStatic --attention-backend triton --sampling-backend pytorch --disable-cuda-graph"
    )
    Invoke-Docker @runArgs
}

Write-Host "Waiting for SGLang readiness: $modelsUrl"
for ($i = 1; $i -le $ReadyChecks; $i++) {
    try {
        $response = Invoke-WebRequest -Uri $modelsUrl -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            Write-Host "SGLang is ready."
            $response.Content
            exit 0
        }
    }
    catch {
        $running = & $docker inspect -f "{{.State.Running}}" $ContainerName 2>$null
        if ($running -ne "true") {
            Write-Host "Container exited before readiness. Last logs:"
            & $docker logs --tail 160 $ContainerName
            exit 1
        }
    }
    Start-Sleep -Seconds 5
}

Write-Host "Timed out waiting for SGLang. Last logs:"
& $docker logs --tail 200 $ContainerName
exit 1
