#!/bin/bash

USER="registeredusers"
PASS="only"

# List your fragment base names here
# fragments=("20240304161941-20240304144031-20240304141531-20231210132040")
# fragments=("20230620230619")
fragments=("20231215151901")

for frag in "${fragments[@]}"; do
    # Base URL
    # base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working/${frag}"
    # base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${frag}"
    # base_url="https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/${frag}"
    base_url="https://dl.ash2txt.org/community-uploads/luke/youssef_uploads/scroll_4/${frag}"
    layers_url="${base_url}/layers/"

    # Create output directory
    out_dir="./train_scrolls/${frag}/layers/"
    mkdir -p "$out_dir"
 
    # Download specific .tif files from x to y slice
    for i in $(seq -w 14 47); do
        wget --user=$USER --password=$PASS \
            -P "$out_dir" \
            "${layers_url}${i}.tif"
    done

    # Download masks
    wget --user=$USER --password=$PASS \
        -P ./train_scrolls/${frag}/ \
        ${base_url}/${frag}_inklabels.png

    wget --user=$USER --password=$PASS \
        -P ./train_scrolls/${frag}/ \
        ${base_url}/${frag}_mask.png
done

# #!/bin/bash

# USER="registeredusers"
# PASS="only"

# # List your fragment base names here
# fragments=("20231122192640")

# for frag in "${fragments[@]}"; do
#     # Base URL
#     base_url="https://dl.ash2txt.org/fragments/${frag}/"
    

#     # Download layers
#     wget --no-parent -r --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/layers/ \
#         --cut-dirs=6 -nH \
#         ${base_url}/layers/

#     # Download masks
#     wget --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/ \
#         ${base_url}/20231122192640_mask.png

#     wget --user=$USER --password=$PASS \
#         -P ./train_scrolls/${frag}/ \
#         ${base_url}/20231122192640_inklabels.png
# done

