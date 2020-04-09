#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

DO_PROJECT_ID="abb6ec4f-4c12-47ec-9ad6-53a9bb9722aa"
DO_REGION="ams3"
DO_SSH_KEY_ID="25709162"
DO_IMAGE_DISTRIBUTION_ID="53893572"
DO_DROPLET_SIZE_SLUG="s-1vcpu-1gb" # smallest, $0.007440 / hr
DO_DROPLET_TAG="big-data"
# DO_DROPLET_SIZE_SLUG="4gb"  # 4GB RAM, 2VCPUs, 60GB Disk, $0.059520 / hr
# DO_DROPLET_SIZE_SLUG="8gb"  # 8GB RAM, 4VCPUs, 80GB Disk, $0.119050 / hr
DO_EXTRA_CREATE_OPTS="--enable-monitoring"

N_SLAVES=2

function status() {
    doctl compute droplet ls --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,PublicIPv4,Image,Memory,VCPUs,Disk"
}

function up() {
    echo "Using droplet image:"
    doctl compute image get "$DO_IMAGE_DISTRIBUTION_ID"
    echo "Using droplet size: $DO_DROPLET_SIZE_SLUG"
    echo "Script will create $((N_SLAVES+1)) droplets"
    
    echo ""
    echo "Creating droplets:"
    # master
    doctl compute droplet create \
        "ubuntu-master" \
        --region "$DO_REGION" \
        --ssh-keys "$DO_SSH_KEY_ID" \
        --image "$DO_IMAGE_DISTRIBUTION_ID" \
        --size "$DO_DROPLET_SIZE_SLUG" \
        --tag-name "$DO_DROPLET_TAG" \
        --format "ID,Name,Image,Memory,VCPUs,Disk" \
        $DO_EXTRA_CREATE_OPTS
    # slaves
    for i in $(seq -f "%02g" 1 $N_SLAVES); do
        doctl compute droplet create \
            "ubuntu-slave-${i}" \
            --region "$DO_REGION" \
            --ssh-keys "$DO_SSH_KEY_ID" \
            --image "$DO_IMAGE_DISTRIBUTION_ID" \
            --size "$DO_DROPLET_SIZE_SLUG" \
            --tag-name "$DO_DROPLET_TAG" \
            --format "ID,Name,Image,Memory,VCPUs,Disk" \
            --no-header \
            $DO_EXTRA_CREATE_OPTS
    done

    # assign to project
    droplets=$(doctl compute droplet ls --format ID --no-header --tag-name "$DO_DROPLET_TAG")
    for droplet in $droplets; do
        doctl projects resources assign "$DO_PROJECT_ID" --resource "do:droplet:$droplet"
    done
    
    # wait for IPs
    echo ""
    echo "Waiting for IP assignment..."
    n_droplets=$(echo "$droplets" | wc -w | tr -d ' ')
    n_ready_droplets=$(doctl compute droplet ls \
        --tag-name "$DO_DROPLET_TAG" \
        --format "PublicIPv4" \
        --no-header | wc -w | tr -d '\blank')
    while [[ "$n_ready_droplets" -lt "$n_droplets" ]]; do
        echo "$n_ready_droplets out of $n_droplets..."
        sleep 3
        n_ready_droplets=$(doctl compute droplet ls \
            --tag-name "$DO_DROPLET_TAG" \
            --format "PublicIPv4" \
            --no-header | wc -w | tr -d '\blank')
    done

    # display status with IPs
    echo ""
    status
}

function down() {
    droplets=$(doctl compute droplet ls --format ID --no-header --tag-name "$DO_DROPLET_TAG")
    doctl compute droplet rm $droplets -f
}

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 [up|down|status]"
    exit 2
fi

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
    echo "Usage: $0 [up|down|status]"
    exit 2
    ;;
esac
