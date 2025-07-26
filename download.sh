#!/bin/bash

USER="registeredusers"
PASS="only"

# List your fragment base names here
fragments=("20240304161941-20240304144031-20240304141531-20231210132040")
# fragments=("20230620230619")

for frag in "${fragments[@]}"; do
    # Base URL
    # base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working/${frag}"
    base_url="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${frag}"
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
        ${base_url}/${frag}_flatboi_mask.png
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


# # scroll1 = http://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/
# # scroll4 = https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/
# # frag5 = https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/

# seg='20240618142020'

# wget --no-parent --user=registeredusers --password=only \
#      -P ./train_scrolls/$seg/ \
#      --cut-dirs=6 -nH \
#      https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240618142020/20240618142020_mask.png
#      # https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240712074250/20240712074250_mask.png
#      # https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/20231122192640/20231122192640_mask.png
#      # https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/20240304141530/20240304141530_mask.png
#      # https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working/working_mesh_0_window_-1511_38488_flatboi_1/mesh_0_window_-1511_38488_flatboi_1_mask.png
# #wget --no-parent --user=registeredusers --password=only \
# #     -P ./train_scrolls/$seg/ \
# #     --cut-dirs=6 -nH \

# #     https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/${seg}/${seg}_mask.png
# wget --no-parent -r --user=registeredusers --password=only \
#      -P ./train_scrolls/$seg/layers/ \
#      --cut-dirs=6 -nH \
#      https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240618142020/layers/
#      # https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240712074250/layers/
#      # https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/20231210132040/layers/
#      # https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working/working_mesh_0_window_-1511_38488_flatboi_1/layers/

# # # wget --no-parent --user=registeredusers --password=only \
# # #      -P ./train_scrolls/$seg/ \
# # #      --cut-dirs=7 -nH \
# # #      https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/mask.png

# #      # https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/mask.png
# #      # https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/mask.png

# #      # https://dl.ash2txt.org/fragments/Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface/mask.png
# #      # https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/mask.png
# #      # https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/PHercParis1Fr39_54keV_mask.png
# # wget --no-parent --user=registeredusers --password=only \
# #      -P ./train_scrolls/$seg/ \
# #      --cut-dirs=7 -nH \
# #      https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/inklabels.png

# #      # https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/inklabels.png
# #      # https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/inklabels.png

# #      # https://dl.ash2txt.org/fragments/Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface/inklabels.png
# #      # https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/PHercParis1Fr39_54keV_inklabels.png	

# # wget --no-parent -r --user=registeredusers --password=only \
# #      -P ./train_scrolls/$seg/layers/ \
# #      --cut-dirs=9 -nH \
# #      https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/surface_volume/

# #      # https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/surface_volume/
# #      # https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/surface_volume/

# #      # https://dl.ash2txt.org/fragments/Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface/surface_volume/
# #      # https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/PHercParis1Fr39_54keV_surface_volume/
