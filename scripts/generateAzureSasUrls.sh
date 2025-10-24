#!/bin/bash
# Script to generate Azure Blob Storage SAS URLs for image files in JSONL format
# requires az cli installed and logged in https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
# Usage: ./generateAzureSasUrls.sh <container-url> [output-file] [expiration-hours] [parallel-jobs]
# Example: ./generateAzureSasUrls.sh https://myaccount.blob.core.windows.net/mycontainer output.jsonl 6 8
# Or with curl:
# curl -fsSL https://raw.githubusercontent.com/roboflow/roboflow-python/main/scripts/generateAzureSasUrls.sh | bash -s -- https://myaccount.blob.core.windows.net/mycontainer output.jsonl

set -e

# Check if container URL is provided
if [ -z "$1" ]; then
    echo "Error: Azure container URL is required"
    echo "Usage: $0 <container-url> [output-file] [expiration-hours] [parallel-jobs]"
    echo "Example: $0 https://myaccount.blob.core.windows.net/mycontainer output.jsonl 6 8"
    exit 1
fi

CONTAINER_URL="$1"
OUTPUT_FILE="${2:-signed_urls.jsonl}"
EXPIRATION_HOURS="${3:-6}"  # Default: 6 hours
PARALLEL_JOBS="${4:-20}"  # Default: 20 parallel jobs

# Remove trailing slash from container URL if present
CONTAINER_URL="${CONTAINER_URL%/}"

# Extract storage account and container from URL
STORAGE_ACCOUNT=$(echo "$CONTAINER_URL" | sed -E 's|https://([^.]+)\.blob\.core\.windows\.net/.*|\1|')
CONTAINER=$(echo "$CONTAINER_URL" | sed -E 's|https://[^/]+/([^/]+).*|\1|')

# Optional: Extract path prefix if provided in URL
PATH_PREFIX=$(echo "$CONTAINER_URL" | sed -E 's|https://[^/]+/[^/]+/?(.*)|/\1|' | sed 's|^//$||')
if [ "$PATH_PREFIX" = "/" ]; then
    PATH_PREFIX=""
fi

# Image file extensions to include (regex pattern for grep)
IMAGE_PATTERN='\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg)$'

# Calculate expiry time in UTC (cross-platform compatible)
if date --version >/dev/null 2>&1; then
    # GNU date (Linux)
    EXPIRY=$(date -u -d "+${EXPIRATION_HOURS} hours" '+%Y-%m-%dT%H:%MZ')
else
    # BSD date (macOS)
    EXPIRY=$(date -u -v+${EXPIRATION_HOURS}H '+%Y-%m-%dT%H:%MZ')
fi

# Function to process a single blob
process_blob() {
    local blob_name="$1"
    local storage_account="$2"
    local container="$3"
    local expiry="$4"
    
    # Generate SAS token for the specific blob (redirect stderr to suppress warnings)
    local sas_token=$(az storage blob generate-sas \
        --account-name "$storage_account" \
        --container-name "$container" \
        --name "$blob_name" \
        --permissions r \
        --expiry "$expiry" \
        --https-only \
        --auth-mode key \
        --output tsv 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        # Construct the full URL with SAS token
        local signed_url="https://${storage_account}.blob.core.windows.net/${container}/${blob_name}?${sas_token}"
        
        # Create name with full path using double underscores instead of slashes
        local name_with_path=$(echo "$blob_name" | sed 's|/|__|g')
        
        # Output JSONL
        echo "{\"name\": \"$name_with_path\", \"url\": \"$signed_url\"}"
    fi
}

# Alternative function using connection string or SAS token at account level
process_blob_with_connection() {
    local blob_name="$1"
    local storage_account="$2"
    local container="$3"
    local expiry="$4"
    
    # Generate SAS token using connection string if AZURE_STORAGE_CONNECTION_STRING is set
    if [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
        local sas_token=$(az storage blob generate-sas \
            --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
            --container-name "$container" \
            --name "$blob_name" \
            --permissions r \
            --expiry "$expiry" \
            --https-only \
            --output tsv 2>/dev/null)
    else
        # Use account key if available
        local sas_token=$(az storage blob generate-sas \
            --account-name "$storage_account" \
            --container-name "$container" \
            --name "$blob_name" \
            --permissions r \
            --expiry "$expiry" \
            --https-only \
            --output tsv 2>/dev/null)
    fi
    
    if [ $? -eq 0 ]; then
        local signed_url="https://${storage_account}.blob.core.windows.net/${container}/${blob_name}?${sas_token}"
        local name_with_path=$(echo "$blob_name" | sed 's|/|__|g')
        echo "{\"name\": \"$name_with_path\", \"url\": \"$signed_url\"}"
    fi
}

# Check if user is logged in to Azure CLI
if ! az account show &>/dev/null; then
    echo "Error: Not logged in to Azure CLI. Please run 'az login' first."
    exit 1
fi

# Export function and variables for xargs
export -f process_blob process_blob_with_connection
export STORAGE_ACCOUNT CONTAINER EXPIRY
export AZURE_STORAGE_CONNECTION_STRING

echo "Listing blobs from container: $CONTAINER in account: $STORAGE_ACCOUNT..."
if [ -n "$PATH_PREFIX" ]; then
    echo "Using path prefix: $PATH_PREFIX"
fi

# Get list of all blobs, filter for images, and process in parallel
# Create a temporary file for the blob list to avoid stdin issues
BLOB_LIST=$(mktemp)
trap "rm -f $BLOB_LIST" EXIT

if [ -n "$PATH_PREFIX" ]; then
    # List blobs with prefix
    az storage blob list \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER" \
        --prefix "$PATH_PREFIX" \
        --auth-mode key \
        --query "[].name" \
        --output tsv 2>/dev/null | grep -iE "$IMAGE_PATTERN" > "$BLOB_LIST"
else
    # List all blobs in container
    az storage blob list \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER" \
        --auth-mode key \
        --query "[].name" \
        --output tsv 2>/dev/null | grep -iE "$IMAGE_PATTERN" > "$BLOB_LIST"
fi

# Process blobs in parallel using background jobs
: > "$OUTPUT_FILE"  # Clear output file
COUNT=0

while IFS= read -r blob_name; do
    # Process blob in background
    process_blob "$blob_name" "$STORAGE_ACCOUNT" "$CONTAINER" "$EXPIRY" >> "$OUTPUT_FILE" &

    # Limit concurrent jobs
    ((COUNT++))
    if [ $((COUNT % PARALLEL_JOBS)) -eq 0 ]; then
        wait  # Wait for current batch to complete
    fi
done < "$BLOB_LIST"

# Wait for any remaining jobs
wait

# Display the results
cat "$OUTPUT_FILE"

echo ""
echo "Done! SAS URLs written to $OUTPUT_FILE"
echo "Total images processed: $(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)"
echo "SAS tokens valid until: $EXPIRY"
