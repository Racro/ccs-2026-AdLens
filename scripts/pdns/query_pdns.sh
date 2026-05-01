#!/bin/bash
set -e

# Usage:

# Step 1: Install zdns
# https://github.com/zmap/zdns#install

# Step 2: Add ad link (landing page URL) domains to all_detected_ad_links.txt.
# Each domain must be on a separate line.

# Step 2: Run
# $ ./query_pdns.sh

run_zdns () {
  local name="$1"
  local resolvers="$2"

  local output="pdns_${name}_ad_links.jsonl"

  zdns A \
    --input-file=all_ad_link_domains.txt \
    --threads=5 \
    --name-servers="$resolvers" \
    > "$output"
}

run_zdns "cloudflare" "1.0.0.2,1.1.1.2,2606:4700:4700::1002,2606:4700:4700::1112"
run_zdns "quad9"      "9.9.9.9,2620:fe::fe,149.112.112.112,2620:fe::9"
run_zdns "cisco"      "208.67.222.222,208.67.220.220,2620:119:53::53,2620:119:35::35"
run_zdns "cira"       "149.112.121.20,149.112.122.20,2620:10A:80BB::20,2620:10A:80BC::20"

