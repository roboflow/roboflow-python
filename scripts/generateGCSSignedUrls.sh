#!/bin/bash

# Script to generate GCS signed URLs for image files in JSONL format
# Usage: ./listgcs.sh <gcs-path> [output-file] [expiration-seconds] [parallel-jobs]

set -e

# Check if GCS path is provided
if [ -z "$1" ]; then
    echo "Error: GCS path is required"
    echo "Usage: $0 <gcs-path> [output-file] [expiration-seconds] [parallel-jobs]"
    echo "Example: $0 gs://my-bucket/images/ output.jsonl 21600 8"
    exit 1
fi

GCS_PATH="$1"
OUTPUT_FILE="${2:-signed_urls.jsonl}"
EXPIRATION_SECONDS="${3:-21600}"  # Default: 6 hours
PARALLEL_JOBS="${4:-20}"  # Default: 20 parallel jobs

# Remove trailing slash from GCS path if present
GCS_PATH="${GCS_PATH%/}"

# Convert seconds to duration format for gcloud (e.g., 21600s)
EXPIRATION="${EXPIRATION_SECONDS}s"

# Image file extensions to include (regex pattern for grep)
IMAGE_PATTERN='\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg)$'

# Function to find an appropriate service account
find_service_account() {
    # First, try to get the default compute service account for the current project
    local project_id=$(gcloud config get-value project 2>/dev/null)
    if [ -n "$project_id" ]; then
        local compute_sa="${project_id}-compute@developer.gserviceaccount.com"
        if gcloud iam service-accounts describe "$compute_sa" >/dev/null 2>&1; then
            echo "$compute_sa"
            return 0
        fi
    fi

    # If that doesn't work, try to find any service account in the project
    local sa_list=$(gcloud iam service-accounts list --format="value(email)" --limit=1 2>/dev/null)
    if [ -n "$sa_list" ]; then
        echo "$sa_list" | head -n 1
        return 0
    fi

    return 1
}

# Try to find a service account to use
SERVICE_ACCOUNT=$(find_service_account)
if [ -z "$SERVICE_ACCOUNT" ]; then
    echo "Warning: No service account found. Attempting to sign URLs without impersonation."
    echo "If this fails, you may need to:"
    echo "1. Authenticate with a service account: gcloud auth activate-service-account --key-file=key.json"
    echo "2. Or ensure you have appropriate service accounts in your project"
    echo ""
fi

# Function to process a single file
process_file() {
    local object="$1"
    local service_account="$2"
    local expiration="$3"

    # Create signed URL using gcloud storage sign-url
    local signed_url_output
    if [ -n "$service_account" ]; then
        signed_url_output=$(gcloud storage sign-url --http-verb=GET --duration="$expiration" --impersonate-service-account="$service_account" "$object" 2>/dev/null)
    else
        signed_url_output=$(gcloud storage sign-url --http-verb=GET --duration="$expiration" "$object" 2>/dev/null)
    fi

    if [ $? -eq 0 ] && [ -n "$signed_url_output" ]; then
        # Extract just the signed_url from the YAML output
        local signed_url=$(echo "$signed_url_output" | grep "signed_url:" | sed 's/signed_url: //')

        if [ -n "$signed_url" ]; then
            # Extract the path after the bucket name and convert slashes to double underscores
            local path_part=$(echo "$object" | sed 's|gs://[^/]*/||')
            local name_with_path=$(echo "$path_part" | sed 's|/|__|g')

            # Output JSONL
            echo "{\"name\": \"$name_with_path\", \"url\": \"$signed_url\"}"
        fi
    fi
}

# Export function and variables for xargs
export -f process_file
export SERVICE_ACCOUNT
export EXPIRATION

echo "Listing files from $GCS_PATH..."

# Get list of all files, filter for images, and process in parallel
gsutil ls -r "$GCS_PATH" 2>/dev/null | \
    grep -v '/$' | \
    grep -v ':$' | \
    grep -iE "$IMAGE_PATTERN" | \
    xargs -I {} -P "$PARALLEL_JOBS" bash -c 'process_file "$@"' _ {} "$SERVICE_ACCOUNT" "$EXPIRATION" | \
    tee "$OUTPUT_FILE"

echo ""
echo "Done! Signed URLs written to $OUTPUT_FILE"
echo "Total images processed: $(wc -l < "$OUTPUT_FILE")"
