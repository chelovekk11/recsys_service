# Smoke test for recsys service (PowerShell)
# Запускать в другом окне, когда контейнер уже поднят
# docker run --rm -p 8000:8000 `
#   -v "${PWD}\recsys_service\artifacts:/srv/artifacts" `
#   --name recsys recsys-mvp:latest

$BASE_URL = "http://127.0.0.1:8000"

Write-Host "GET /health" -ForegroundColor Cyan
Invoke-WebRequest "$BASE_URL/health" | Select-Object -Expand Content

Write-Host "`nGET /metrics" -ForegroundColor Cyan
Invoke-WebRequest "$BASE_URL/metrics" | Select-Object -Expand Content

Write-Host "`nPOST /predict (valid)" -ForegroundColor Cyan
$body_ok = @{
  user_id = 123
  recent_items = @(119736, 7943, 248676)
  k = 3
} | ConvertTo-Json
Invoke-WebRequest -Uri "$BASE_URL/predict" -Method POST `
  -Headers @{ "Content-Type" = "application/json" } -Body $body_ok `
| Select-Object -Expand Content

Write-Host "`nPOST /predict (invalid: k=0 -> expect 422)" -ForegroundColor Cyan
try {
  $body_bad = @{ user_id=123; recent_items=@(119736); k=0 } | ConvertTo-Json
  Invoke-WebRequest -Uri "$BASE_URL/predict" -Method POST `
    -Headers @{ "Content-Type"="application/json" } `
    -Body $body_bad | Out-Null
} catch {
  $resp = $_.Exception.Response
  if ($resp) {
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $reader.ReadToEnd() | Write-Output
  } else {
    Write-Error $_
  }
}
