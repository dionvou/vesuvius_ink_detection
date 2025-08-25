# #!/bin/bash

# USER="registeredusers"
# PASS="only"

# # List your fragments
# fragments=("20240618142020")

# # Loop through each fragment
# for frag in "${fragments[@]}"; do
#     base_url="https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/${frag}"
#     layers_url="${base_url}/layers/"
    
#     # Check layers 15 to 45 (try both .tif and .png)
#     for i in $(seq -w 15 45); do
#         found=false
#         for ext in tif png; do
#             url="${layers_url}${i}.${ext}"
#             wget --user=$USER --password=$PASS --spider --quiet "$url"
#             if [ $? -eq 0 ]; then
#                 echo "Exists: ${url}"
#                 found=true
#                 break
#             fi
#         done
#         if [ "$found" = false ]; then
#             echo "Missing: Slice ${i} not found in .tif or .png for ${frag}"
#         fi
#     done

#     # Check masks and inklabels
#     mask_url="${base_url}/${frag}_mask.png"
#     wget --user=$USER --password=$PASS --spider --quiet "$mask_url"
#     if [ $? -eq 0 ]; then
#         echo "Exists: ${mask_url}"
#     else
#         echo "Missing: Mask not found for ${frag}"
#     fi

#     # inklabels check (currently commented out in your script)
#     ink_url="${base_url}/${frag}_inklabels.png"
#     wget --user=$USER --password=$PASS --spider --quiet "$ink_url"
#     if [ $? -eq 0 ]; then
#         echo "Exists: ${ink_url}"
#     else
#         echo "Missing: Inklabels not found for ${frag}"
#     fi
# done
#!/bin/bash

# USER="registeredusers"
# PASS="only"

# # Define fragments with their corresponding base URLs
# # Format: [fragment]=base_url
# declare -A fragments
# fragments=(
#     # ["20230620230619"]="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/thaumato_outputs/scroll4_thaumato_mar17/working"
#     ["20231210132040"]="https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths"
#     ["20240618142020"]="https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths"
# )

# # Loop through each fragment
# for frag in "${!fragments[@]}"; do
#     base_url="${fragments[$frag]}/${frag}"
#     layers_url="${base_url}/layers/"
    
#     echo "Checking fragment: $frag"
    
#     # Check layers 15 to 45 (try both .tif and .png)
#     for i in $(seq -w 15 45); do
#         found=false
#         for ext in tif png; do
#             url="${layers_url}${i}.${ext}"
#             wget --user=$USER --password=$PASS --spider --quiet "$url"
#             if [ $? -eq 0 ]; then
#                 echo "Exists: ${url}"
#                 found=true
#                 break
#             fi
#         done
#         if [ "$found" = false ]; then
#             echo "Missing: Slice ${i} not found in .tif or .png for ${frag}"
#         fi
#     done

#     # Check mask
#     mask_url="${base_url}/${frag}_mask.png"
#     wget --user=$USER --password=$PASS --spider --quiet "$mask_url"
#     if [ $? -eq 0 ]; then
#         echo "Exists: ${mask_url}"
#     else
#         echo "Missing: Mask not found for ${frag}"
#     fi

#     # Check inklabels
#     ink_url="${base_url}/${frag}_inklabels.png"
#     wget --user=$USER --password=$PASS --spider --quiet "$ink_url"
#     if [ $? -eq 0 ]; then
#         echo "Exists: ${ink_url}"
#     else
#         echo "Missing: Inklabels not found for ${frag}"
#     fi

#     echo "----------------------------------------"
# done
#!/bin/bash

#!/bin/bash

# USER="registeredusers"
# PASS="only"

# # Base URLs
# urls=(
#     "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths"
#     "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths"
#     "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths"
#     "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths"
# )

# echo "Searching for fragments..."

# for base_url in "${urls[@]}"; do
#     echo "Checking $base_url"

#     # Attempt to get directory listing
#     frags=$(curl -s --user "$USER:$PASS" "$base_url/" | grep -oP 'href="\K[^"/]+(?=/")')

#     if [ -z "$frags" ]; then
#         echo "No fragments found or directory listing not allowed for $base_url"
#     else
#         echo "Fragments found:"
#         echo "$frags"
#     fi

#     echo "----------------------------------------"
# done

#!/bin/bash
#!/bin/bash

USER="registeredusers"
PASS="only"

# Base URLs
urls=(
    "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths"
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths"
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths"
    "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths"
)


for base_url in "${urls[@]}"; do
    echo "Checking base URL: $base_url"

    # Attempt to get fragments
    frags=$(curl -s --user "$USER:$PASS" "$base_url/" | grep -oP 'href="\K[^"/]+(?=/")')

    if [ -z "$frags" ]; then
        echo "No fragments found at $base_url or directory listing not allowed."
        continue
    fi

    for frag in $frags; do
        slice_found=false
        mask_found=false

        # Check layer 32
        layers_url="${base_url}/${frag}/layers/"
        for ext in tif png jpg; do
            file_url="${layers_url}32.${ext}"
            wget --user="$USER" --password="$PASS" --spider --quiet "$file_url"
            if [ $? -eq 0 ]; then
                slice_found=true
                break
            fi
        done

        # # Check mask file
        # for ext in png jpg; do
        #     mask_url="${base_url}/${frag}/${frag}_mask.${ext}"
        #     wget --user="$USER" --password="$PASS" --spider --quiet "$mask_url"
        #     if [ $? -eq 0 ]; then
        #         mask_found=true
        #         break
        #     fi
        # done
        # Generic mask check: any file ending with _mask.png or _mask.jpg
        mask_found=false

        # Try to list fragment directory
        frag_listing=$(curl -s --user "$USER:$PASS" "$base_url/$frag/")

        if [ -n "$frag_listing" ]; then
            # Look for anything ending with _mask.png or _mask.jpg
            if echo "$frag_listing" | grep -qE 'href="[^"]+mask\.(png|jpg)"'; then
                mask_found=true
            fi
        fi

        # Summarize results
        if [ "$slice_found" = true ] && [ "$mask_found" = true ]; then
            status="OK ✅"
        elif [ "$slice_found" = true ] || [ "$mask_found" = true ]; then
            status="Partial ⚠"
        else
            status="NOT OK ❌"
        fi

        echo "Fragment: $frag | Slice 32: $slice_found | Mask: $mask_found | Status: $status"
        # echo "
        # ----------------------------------------"
    done
done