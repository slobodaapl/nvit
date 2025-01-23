#!/bin/bash

num_gpus=1
visible_gpus="all"
profiles_dir="profiles"

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        --visible_gpus)
            visible_gpus="$2"
            shift 2
            ;;
        --profiles-dir)
            profiles_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

env_files=("$profiles_dir"/*.env)
if [ ${#env_files[@]} -eq 0 ] || [ ! -e "${env_files[0]}" ]; then
    echo "No .env files found in $profiles_dir directory"
    exit 1
fi

# Read .env file if it exists and prepend it to the profile env files
local_env=""
if [ -f ".env" ]; then
    local_env=$(cat .env)
fi

for env_file in "${env_files[@]}"; do
    echo "Profile: $env_file"
    profile_env=$(cat "$env_file")

    ./docker_launcher.sh \
        --num_gpus "$num_gpus" \
        --visible_gpus "$visible_gpus" \
        --env-file <(echo -e "${local_env}\n${profile_env}")
done