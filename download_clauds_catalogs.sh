#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
catalog_dir="${script_dir}/data/clauds/catalogs"

mkdir -p "${catalog_dir}"

download_if_missing() {
  local url="$1"
  local filename="$2"
  local destination="${catalog_dir}/${filename}"
  local partial="${destination}.part"

  if [[ -f "${destination}" ]]; then
    echo "exists: ${destination}"
    return 0
  fi

  echo "download: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --output "${partial}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${partial}" "${url}"
  else
    echo "error: neither curl nor wget is available" >&2
    return 1
  fi

  mv "${partial}" "${destination}"
}

download_if_missing \
  "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/clauds/desprez/PublicRelease/COSMOS-HSCpipe-Phosphoros.fits" \
  "COSMOS-HSCpipe-Phosphoros.fits"

download_if_missing \
  "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/clauds/desprez/PublicRelease/DEEP23-HSCpipe-Phosphoros.fits" \
  "DEEP23-HSCpipe-Phosphoros.fits"
