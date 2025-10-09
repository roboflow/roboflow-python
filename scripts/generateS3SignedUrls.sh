#!/bin/bash

# Script to generate S3 signed URLs for image files in JSONL format
# Usage: ./generateS3SignedUrls.sh <s3-path> [output-file] [expiration-seconds] [parallel-jobs]
# Or with curl:
# curl -fsSL https://gist.githubusercontent.com/tonylampada/20b7bc984a455f53e2d07f88b33bf43c/raw/generateS3SignedUrls.sh | bash -s --  s3://bucket/path output.jsonl

set -e

# Check if S3 path is provided
if [ -z "$1" ]; then
    echo "Error: S3 path is required"
    echo "Usage: $0 <s3-path> [output-file] [expiration-seconds] [parallel-jobs]"
    echo "Example: $0 s3://my-bucket/images/ output.jsonl 3600 8"
    exit 1
fi

S3_PATH="$1"
OUTPUT_FILE="${2:-signed_urls.jsonl}"
EXPIRATION="${3:-21600}"  # Default: 6 hours
PARALLEL_JOBS="${4:-20}"  # Default: 20 parallel jobs

# Remove trailing slash from S3 path if present
S3_PATH="${S3_PATH%/}"

# Extract bucket name from S3_PATH
BUCKET=$(echo "$S3_PATH" | sed 's|s3://||' | cut -d'/' -f1)

# Image file extensions to include (regex pattern for grep)
IMAGE_PATTERN='\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg)$'

# Function to process a single file
process_file() {
    local file_path="$1"
    local bucket="$2"
    local expiration="$3"

    # Construct full S3 URI
    local s3_uri="s3://${bucket}/${file_path}"

    # Generate signed URL
    local signed_url=$(aws s3 presign "$s3_uri" --expires-in "$expiration" 2>/dev/null)

    if [ $? -eq 0 ]; then
        # Create name with full path using double underscores instead of slashes
        local name_with_path=$(echo "$file_path" | sed 's|/|__|g')

        # Output JSONL
        echo "{\"name\": \"$name_with_path\", \"url\": \"$signed_url\"}"
    fi
}

# Export function and variables for xargs
export -f process_file
export BUCKET
export EXPIRATION

echo "Listing files from $S3_PATH..."

# Get list of all files, filter for images, and process in parallel
aws s3 ls "$S3_PATH/" --recursive | \
    awk '{print $4}' | \
    grep -iE "$IMAGE_PATTERN" | \
    xargs -I {} -P "$PARALLEL_JOBS" bash -c 'process_file "$@"' _ {} "$BUCKET" "$EXPIRATION" | \
    tee "$OUTPUT_FILE"

echo ""
echo "Done! Signed URLs written to $OUTPUT_FILE"
echo "Total images processed: $(wc -l < "$OUTPUT_FILE")"