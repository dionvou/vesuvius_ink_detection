#!/usr/bin/env bash
set -euo pipefail

USER="registeredusers"
PASS="only"

# Base URLs
urls=(
    "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths"
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths"
    "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths"
)

# Create the parent folder
mkdir -p pretraining_scrolls

for base_url in "${urls[@]}"; do
    echo "🔍 Checking base URL: $base_url"

    # Attempt to get fragments
    frags=$(curl -s --user "$USER:$PASS" "$base_url/" \
        | grep -oP 'href="\K[^"/]+(?=/")')

    if [ -z "$frags" ]; then
        echo "⚠️  No fragments found at $base_url"
        continue
    fi

    for frag in $frags; do
        slice_found=false
        mask_found=false

        # --- Check for layer 32 existence ---
        layers_url="${base_url}/${frag}/layers/"
        for ext in tif tiff png jpg; do
            file_url="${layers_url}32.${ext}"
            if wget --user="$USER" --password="$PASS" --spider --quiet "$file_url"; then
                slice_found=true
                break
            fi
        done

        # --- Check for mask existence ---
        frag_listing=$(curl -s --user "$USER:$PASS" "$base_url/$frag/")
        if echo "$frag_listing" | grep -qE 'href="[^"]+mask\.(png|jpg)"'; then
            mask_found=true
        fi

        # --- If both exist, download ---
        if [ "$slice_found" = true ] && [ "$mask_found" = true ]; then
            echo "✅ Fragment: $frag | Downloading layers + mask"

            outdir="pretraining_scrolls/$frag"
            mkdir -p "$outdir"

            # Download mask file(s)
            mask_files=$(echo "$frag_listing" | grep -oP 'href="\K[^"]*mask\.(png|jpg)')
            for mask in $mask_files; do
                echo "    ⬇️  Mask: $mask"
                wget --quiet --show-progress \
                    --user="$USER" --password="$PASS" \
                    -P "$outdir" "$base_url/$frag/$mask"
            done

            # Download layers/ folder
            echo "    ⬇️  Layers/"
            wget --quiet --show-progress -r -np -nH --cut-dirs=5 \
                --user="$USER" --password="$PASS" \
                -P "$outdir" \
                "$layers_url"

        else
            echo "❌ Fragment: $frag | Slice 32: $slice_found | Mask: $mask_found | Skipping download"
        fi
    done
done

echo "🎉 Done. Valid fragments (mask + slice 32) downloaded into pretraining_scrolls/."
