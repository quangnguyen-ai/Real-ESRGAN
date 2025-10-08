#!/bin/bash

# Script to extract nested OST dataset zip files and organize into DF2K/OST
# Usage: bash extract_ost.sh

set -e  # Exit on error

echo "=== Extracting OST Dataset ==="

# Create target directory
echo "Step 1: Creating target directory..."
mkdir -p DF2K/OST

# Extract each category zip and move images
echo "Step 2: Extracting category zip files..."
cd OutdoorSceneTrain_v2

for category_zip in *.zip; do
    if [ -f "$category_zip" ]; then
        category_name="${category_zip%.zip}"
        echo "  Processing $category_name..."
        
        # Extract to temporary directory
        unzip -q -o "$category_zip" -d "temp_$category_name"
        
        # Move all images to OST directory (flatten structure)
        find "temp_$category_name" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} ../DF2K/OST/ \;
        
        # Clean up temp directory
        rm -rf "temp_$category_name"
    fi
done

cd ..

# Count images
image_count=$(ls -1 DF2K/OST | wc -l)
echo ""
echo "=== Extraction Complete ==="
echo "Total images in DF2K/OST: $image_count"
echo ""
echo "Next steps:"
echo "1. Verify image count (~10,324 images expected)"
echo "2. Run: bash create_subset.sh"
