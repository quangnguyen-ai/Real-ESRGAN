#!/bin/bash

# Script to create dataset subset for training
# Usage: bash create_subset.sh

set -e

echo "=== Creating Dataset Subset ==="

# Create subset directories
echo "Step 1: Creating subset directories..."
mkdir -p DF2K_subset/DIV2K_train_HR
mkdir -p DF2K_subset/Flickr2K_HR
mkdir -p DF2K_subset/OST

# Copy all DIV2K images
echo "Step 2: Copying DIV2K images (800 images)..."
cp -r DF2K/DIV2K_train_HR/* DF2K_subset/DIV2K_train_HR/
div2k_count=$(ls -1 DF2K_subset/DIV2K_train_HR | wc -l)
echo "  Copied: $div2k_count images"

# Sample 1500 images from Flickr2K
echo "Step 3: Sampling Flickr2K images (1500 images)..."
cd DF2K/Flickr2K_HR
ls | shuf -n 1500 | xargs -I {} cp {} ../../DF2K_subset/Flickr2K_HR/
cd ../..
flickr_count=$(ls -1 DF2K_subset/Flickr2K_HR | wc -l)
echo "  Sampled: $flickr_count images"

# Sample 1000 images from OST
echo "Step 4: Sampling OST images (1000 images)..."
cd DF2K/OST
ls | shuf -n 1000 | xargs -I {} cp {} ../../DF2K_subset/OST/
cd ../..
ost_count=$(ls -1 DF2K_subset/OST | wc -l)
echo "  Sampled: $ost_count images"

# Calculate total
total=$((div2k_count + flickr_count + ost_count))

echo ""
echo "=== Subset Creation Complete ==="
echo "DIV2K:    $div2k_count images"
echo "Flickr2K: $flickr_count images"
echo "OST:      $ost_count images"
echo "Total:    $total images"
echo ""
echo "Next step:"
echo "python scripts/generate_meta_info.py \\"
echo "  --input datasets/DF2K_subset/DIV2K_train_HR datasets/DF2K_subset/Flickr2K_HR datasets/DF2K_subset/OST \\"
echo "  --root datasets/DF2K_subset datasets/DF2K_subset datasets/DF2K_subset \\"
echo "  --meta_info datasets/DF2K_subset/meta_info/meta_info_DF2K_subset.txt"
