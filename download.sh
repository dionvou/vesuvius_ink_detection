# # #!/bin/bash

# # USER="registeredusers"
# # PASS="only"

# # # List your fragment base names here
# # # fragments=("20240304161941-20240304144031-20240304141531-20231210132040")
# # # fragments=("20230620230619")
# # fragments=("20231215151901")

# # for frag in "${fragments[@]}"; do
# #     # Base URL
# #     # base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working/${frag}"
# #     # base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${frag}"
# #     # base_url="https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/${frag}"
# #     base_url="https://dl.ash2txt.org/community-uploads/luke/youssef_uploads/scroll_4/${frag}"
# #     layers_url="${base_url}/layers/"

# #     # Create output directory
# #     out_dir="./train_scrolls/${frag}/layers/"
# #     mkdir -p "$out_dir"
 
# #     # Download specific .tif files from x to y slice
# #     for i in $(seq -w 14 47); do
# #         wget --user=$USER --password=$PASS \
# #             -P "$out_dir" \
# #             "${layers_url}${i}.tif"
# #     done

# #     # Download masks
# #     wget --user=$USER --password=$PASS \
# #         -P ./train_scrolls/${frag}/ \
# #         ${base_url}/${frag}_inklabels.png

# #     wget --user=$USER --password=$PASS \
# #         -P ./train_scrolls/${frag}/ \
# #         ${base_url}/${frag}_mask.png
# # done

# # # #!/bin/bash

# # # USER="registeredusers"
# # # PASS="only"

# # # # List your fragment base names here
# # # fragments=("20231122192640")

# # # for frag in "${fragments[@]}"; do
# # #     # Base URL
# # #     base_url="https://dl.ash2txt.org/fragments/${frag}/"
    

# # #     # Download layers
# # #     wget --no-parent -r --user=$USER --password=$PASS \
# # #         -P ./train_scrolls/${frag}/layers/ \
# # #         --cut-dirs=6 -nH \
# # #         ${base_url}/layers/

# # #     # Download masks
# # #     wget --user=$USER --password=$PASS \
# # #         -P ./train_scrolls/${frag}/ \
# # #         ${base_url}/20231122192640_mask.png

# # #     wget --user=$USER --password=$PASS \
# # #         -P ./train_scrolls/${frag}/ \
# # #         ${base_url}/20231122192640_inklabels.png
# # # done

# #!/bin/bash

# USER="registeredusers"
# PASS="only"

# # List your fragments
# fragments=("Frag2" "Frag3" "Frag4")

# # Loop through each fragment
# for frag in "${fragments[@]}"; do
#     base_url="https://dl.ash2txt.org/community-uploads/jrudolph/rescaled-fragments/train_scrolls/${frag}"
#     layers_url="${base_url}/layers/"
    
#     # Create output directory
#     out_dir="./train_scrolls/${frag}/layers/"
#     mkdir -p "$out_dir"

#     # Download layers 15 to 45 (try both .tif and .png)
#     for i in $(seq -w 15 45); do
#         found=false
#         for ext in tif png; do
#             url="${layers_url}${i}.${ext}"
#             wget --user=$USER --password=$PASS \
#                 --spider --quiet "$url"
#             if [ $? -eq 0 ]; then
#                 wget --user=$USER --password=$PASS \
#                     -P "$out_dir" "$url"
#                 found=true
#                 break
#             fi
#         done
#         if [ "$found" = false ]; then
#             echo "Warning: Slice ${i} not found in .tif or .png for ${frag}"
#         fi
#     done

#     # Download masks and inklabels
#     wget --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/ \
#         "${base_url}/${frag}_mask.png"

#     wget --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/ \
#         "${base_url}/${frag}_inklabels.png"
# done

# USER="registeredusers"
# PASS="only"

# # List of fragments to download
# fragments=("20231210132040,")

# # Loop through each fragment
# for frag in "${fragments[@]}"; do

#     base_url="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface//${frag}"
#     layers_url="${base_url}/layers/"
    
#     # Create output directory
#     out_dir="./train_scrolls/${frag}/layers/"
#     mkdir -p "$out_dir"

#     # Download layers 15 to 45 (try both .tif and .png)
#     for i in $(seq -w 1 64); do
#         found=false
#         for ext in tif png jpg; do
#             url="${layers_url}${i}.${ext}"
#             wget --user=$USER --password=$PASS \
#                 --spider --quiet "$url"
#             if [ $? -eq 0 ]; then
#                 wget --user=$USER --password=$PASS \
#                     -P "$out_dir" "$url"
#                 found=true
#                 break
#             fi
#         done
#         if [ "$found" = false ]; then
#             echo "Warning: Slice ${i} not found in .tif or .png for ${frag}"
#         fi
#     done

#     # Download masks and inklabels
#     wget --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/ \
#         "${base_url}/${frag}_mask.png"

#     # wget --user=$USER --password=$PASS \
#     #     -P ./train_scrolls/${frag}/ \
#     #     "${base_url}/${frag}_inklabels.png"
# done

#!/bin/bash

# ================================
# Credentials
# ================================
USER="registeredusers"
PASS="only"

# ================================
# Functions
# ================================

download_layers () {
    local base_url="$1"
    local out_dir="$2"
    local start="$3"
    local end="$4"
    local extensions=("${!5}")

    mkdir -p "$out_dir"

    for i in $(seq -w "$start" "$end"); do
        local found=false
        for ext in "${extensions[@]}"; do
            local url="${base_url}${i}.${ext}"

            # Check existence
            if wget --user="$USER" --password="$PASS" --spider --quiet "$url"; then
                wget --user="$USER" --password="$PASS" -P "$out_dir" "$url"
                found=true
                break
            fi
        done

        if [ "$found" = false ]; then
            echo "Warning: Slice ${i} not found for: ${base_url}"
        fi
    done
}

download_aux_files () {
    local base_url="$1"
    local frag="$2"
    local out_dir="$3"

    wget --user="$USER" --password="$PASS" -P "$out_dir" \
        "${base_url}/${frag}_mask.png"

    # Optional inklabels
    if wget --user="$USER" --password="$PASS" --spider --quiet \
        "${base_url}/${frag}_inklabels.png"; then
        wget --user="$USER" --password="$PASS" -P "$out_dir" \
            "${base_url}/${frag}_inklabels.png"
    fi
}

# ================================
# First batch: Frags
# ================================

fragments=("Frag1")
extensions1=(tif png)

for frag in "${fragments[@]}"; do
    base_url="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/${frag}"
    layers_url="${base_url}/layers/"
    out_dir="./train_scrolls/${frag}/layers/"

    download_layers "$layers_url" "$out_dir" 15 45 extensions1[@]
    download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
done

fragments=("Frag5")
extensions1=(tif png)
for frag in "${fragments[@]}"; do

    base_url="https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/${frag}"
    layers_url="${base_url}/layers/"
    out_dir="./train_scrolls/${frag}/layers/"

    download_layers "$layers_url" "$out_dir" 15 45 extensions1[@]
    download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
done

# ================================
# Second batch: full scrolls
# ================================

fragments2=("20231210132040")
extensions2=(tif png jpg)

for frag in "${fragments2[@]}"; do

    base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${frag}"
    layers_url="${base_url}/layers/"
    out_dir="./train_scrolls/${frag}/layers/"

    download_layers "$layers_url" "$out_dir" 1 64 extensions2[@]
    download_aux_files "$base_url" "$frag" "./train_scrolls/${frag}/"
done
