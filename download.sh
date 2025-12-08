#!/bin/bash

# ================================
# Credentials
# ================================
# Authentication credentials for the Vesuvius Challenge data repository
USER=#"registeredusers"
PASS=#"only"

# ================================
# Functions
# ================================

# Function to download a range of layer files with different possible extensions
# Parameters:
#   $1 - base_url: Base URL path for the layer files
#   $2 - out_dir: Output directory where files will be saved
#   $3 - start: Starting layer number
#   $4 - end: Ending layer number
#   $5 - extensions: Array reference of file extensions to try (e.g., tif, png)
download_layers () {
    local base_url="$1"
    local out_dir="$2"
    local start="$3"
    local end="$4"
    local extensions=("${!5}")

    # Create output directory if it doesn't exist
    mkdir -p "$out_dir"

    # Loop through each layer number (zero-padded sequence)
    for i in $(seq -w "$start" "$end"); do
        local found=false
        # Try each extension until we find one that exists
        for ext in "${extensions[@]}"; do
            local url="${base_url}${i}.${ext}"

            # Check if the file exists on the server without downloading
            if wget --user="$USER" --password="$PASS" --spider --quiet "$url"; then
                # File exists, download it to the output directory
                wget --user="$USER" --password="$PASS" -P "$out_dir" "$url"
                found=true
                break
            fi
        done

        # Warn if no file with any extension was found for this layer
        if [ "$found" = false ]; then
            echo "Warning: Slice ${i} not found for: ${base_url}"
        fi
    done
}

# Function to download auxiliary files (masks and ink labels) for a fragment
# Parameters:
#   $1 - base_url: Base URL path for the auxiliary files
#   $2 - frag: Fragment name/identifier (used in output filename)
#   $3 - out_dir: Output directory where files will be saved
download_aux_files () {
    local base_url="$1"
    local frag="$2"
    local out_dir="$3"

    # Create output directory if it doesn't exist
    mkdir -p "$out_dir"

    # File suffixes to look for (mask images and ink label annotations)
    suffixes=("mask" "inklabels")
    # Possible file extensions to check
    exts=(png jpg tif)

    # Try each suffix (mask, inklabels)
    for suf in "${suffixes[@]}"; do
        # Try each extension (png, jpg, tif)
        for ext in "${exts[@]}"; do
            # We don't know the full filename prefix, so use the directory listing
            local url="${base_url}"
            local pattern=".*${suf}\.${ext}"

            # Get directory listing from the server
            files=$(wget --user="$USER" --password="$PASS" -qO- "$url/" \
                | grep -oP 'href="[^"]+"' | cut -d'"' -f2)

            # Search through the directory listing for matching files
            for f in $files; do
                # Check if filename matches the pattern (ends with suffix.extension)
                if [[ "$f" =~ ${suf}\.${ext}$ ]]; then
                    echo "Found: $f"
                    # Download and rename to standardized format: {frag}_{suffix}.{ext}
                    wget --user="$USER" --password="$PASS" \
                         -O "${out_dir}/${frag}_${suf}.${ext}" \
                         "${url}/${f}"
                    break 2   # Stop after downloading the first matched file
                fi
            done
        done
    done
}

# # ================================
# # First batch: Fragment downloads
# # ================================
# # Download individual papyrus fragments with known ink labels for training

# # Fragment 1: PHercParis2Fr47 scanned at 54keV with 3.24um resolution
# fragments=("Frag1")
# extensions1=(tif png)

# for frag in "${fragments[@]}"; do
#     # Base URL for Fragment 1 data
#     base_url="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface"
#     layers_url="${base_url}/surface_volume/"
#     out_dir="./train_scrolls/${frag}/layers/"

#     # Download layers 15-45 and auxiliary files (mask, inklabels)
#     download_layers "$layers_url" "$out_dir" 15 45 extensions1[@]
#     download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
# done

# # Fragment 5: PHerc1667Cr1Fr3 scanned at 70keV with 3.24um resolution
# fragments=("Frag5")
# extensions1=(tif png)
# for frag in "${fragments[@]}"; do

#     # Base URL for Fragment 5 data
#     base_url="https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing"
#     layers_url="${base_url}/surface_volume/"
#     out_dir="./train_scrolls/${frag}/layers/"

#     # Download layers 15-45 and auxiliary files (mask, inklabels)
#     download_layers "$layers_url" "$out_dir" 15 45 extensions1[@]
#     download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
# done

# # ================================
# # Second batch: Full scroll downloads
# # ================================
# # Download full scroll data (Scroll 4 - PHerc1667)
# # These are larger intact scrolls rather than small fragments

# # Scroll 4 segment
# fragments2=("20231210132040")
# extensions2=(tif png jpg)

# for frag in "${fragments2[@]}"; do

#     # Base URL for Scroll 4, specific path/segmentation
#     base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${frag}"
#     layers_url="${base_url}/layers/"
#     out_dir="./train_scrolls/${frag}/layers/"

#     # Download layers 15-45 and auxiliary files (mask, inklabels)
#     download_layers "$layers_url" "$out_dir" 15 45 extensions2[@]
#     download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
# done

# Scroll 4 segment big
fragments2=("big20231210132040")
extensions2=(tif png jpg)

for frag in "${fragments2[@]}"; do

    # Base URL for Scroll 4, specific path/segmentation
    base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/20231210132040/render_3.24um_20231107190228/"
    layers_url="${base_url}/layers_jpg/"
    out_dir="./train_scrolls/${frag}/layers/"

    # Download layers 15-45 and auxiliary files (mask, inklabels)
    download_layers "$layers_url" "$out_dir" 15 45 extensions2[@]
    download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
done
