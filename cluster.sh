#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

DO_PROJECT_ID="abb6ec4f-4c12-47ec-9ad6-53a9bb9722aa"
DO_REGION="ams3"
DO_IMAGE_DISTRIBUTION_ID="53893572"
DO_DROPLET_SIZE_SLUG="s-1vcpu-1gb" # smallest, $0.007440 / hr
DO_DROPLET_TAG="big-data"
# DO_DROPLET_SIZE_SLUG="4gb"  # 4GB RAM, 2VCPUs, 60GB Disk, $0.059520 / hr
# DO_DROPLET_SIZE_SLUG="8gb"  # 8GB RAM, 4VCPUs, 80GB Disk, $0.119050 / hr
DO_EXTRA_CREATE_OPTS="--enable-monitoring"

function status() {
    doctl compute droplet ls --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,PublicIPv4,Image,Memory,VCPUs,Disk"
}

function up() {
    echo "Using droplet image:"
    doctl compute image get "$DO_IMAGE_DISTRIBUTION_ID"
    echo "Using droplet size: $DO_DROPLET_SIZE_SLUG"
    
    echo ""
    echo "Creating droplets:"
    doctl compute droplet create \
        "ubuntu-master" \
        --region "$DO_REGION" \
        --image "$DO_IMAGE_DISTRIBUTION_ID" \
        --size "$DO_DROPLET_SIZE_SLUG" \
        --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,Image,Memory,VCPUs,Disk" \
        $DO_EXTRA_CREATE_OPTS
    doctl compute droplet create \
        "ubuntu-slave-01" \
        --region "$DO_REGION" \
        --image "$DO_IMAGE_DISTRIBUTION_ID" \
        --size "$DO_DROPLET_SIZE_SLUG" \
        --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,Image,Memory,VCPUs,Disk" \
        --no-header \
        $DO_EXTRA_CREATE_OPTS
    doctl compute droplet create \
        "ubuntu-slave-02" \
        --region "$DO_REGION" \
        --image "$DO_IMAGE_DISTRIBUTION_ID" \
        --size "$DO_DROPLET_SIZE_SLUG" \
        --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,Image,Memory,VCPUs,Disk" \
        --no-header \
        $DO_EXTRA_CREATE_OPTS

    droplets=$(doctl compute droplet ls --format ID --no-header --tag-name "$DO_DROPLET_TAG")
    for droplet in $droplets; do
        doctl projects resources assign "$DO_PROJECT_ID" --resource "do:droplet:$droplet"
    done
    
    while [[ -z "$(doctl compute droplet ls --tag-name "$DO_DROPLET_TAG" --format "PublicIPv4" --no-header)" ]]; do
        sleep 2
    done
    echo ""
    status
}

function down() {
    doctl compute droplet rm "ubuntu-master" "ubuntu-slave-01" "ubuntu-slave-02" -f
}

case "$1" in
up)
    up
    ;;
status)
    status
    ;;
down)
    down
    ;;
*)
    echo "$1 [up|down|status]"
    exit 2
    ;;
esac
