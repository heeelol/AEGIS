# Rename datasets to consistent naming convention
$base_path = "C:\Users\yapor\OneDrive\Desktop\CDE3301\TEnterns\aegis-core\models\data"

# Define rename mappings
$renames = @{
    "Bin identification.coco-segmentation" = "dataset_01_roboflow";
    "AEGIS.coco-segmentation" = "dataset_02_aegis";
    "Bin identification.coco-segmentation (1)" = "dataset_03_bin_identification"
}

Write-Host "Renaming datasets to consistent naming convention..." -ForegroundColor Cyan
Write-Host ""

foreach ($old_name in $renames.Keys) {
    $new_name = $renames[$old_name]
    $old_path = Join-Path $base_path $old_name
    $new_path = Join-Path $base_path $new_name
    
    if (Test-Path $old_path) {
        Write-Host "  Renaming: '$old_name'" -ForegroundColor Yellow
        Write-Host "        -> '$new_name'" -ForegroundColor Green
        Rename-Item -Path $old_path -NewName $new_name -Force
        Write-Host "  OK" -ForegroundColor Green
        Write-Host ""
    }
    else {
        Write-Host "  Not found: $old_path" -ForegroundColor Red
        Write-Host ""
    }
}

Write-Host "All datasets renamed!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Final structure:" -ForegroundColor Cyan
Get-ChildItem -Path $base_path -Directory | Where-Object {$_.Name -like "dataset_*"} | ForEach-Object {
    Write-Host "  * $($_.Name)"
}
