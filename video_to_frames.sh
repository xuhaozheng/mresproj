#!/usr/bin/env zsh


setopt nullglob

source_dirs=("data/video/meningioma" "data/video/GBM")
target_root="data/image"
file_types=("mpg" "avi")  

for dir in "${source_dirs[@]}"; do
    for type in "${file_types[@]}"; do
        for video in "$dir"/*.${type}; do
            if [ -f "$video" ]; then
                filename=$(basename "$video" | cut -d. -f1)
                mkdir -p "$target_root/$filename"
                ffmpeg -i "$video" -vf "fps=24" "$target_root/$filename/%d.png"
            fi
        done
    done
done


unsetopt nullglob


