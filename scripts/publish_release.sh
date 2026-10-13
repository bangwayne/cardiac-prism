#!/usr/bin/env bash
# Resume an interrupted draft upload; never replace published release assets.
set -euo pipefail
release_tag="${1:?Release tag required}"
asset_folder="$(realpath "${2:?Verified asset folder required}")"
release_notes="${3:?Release notes required}"
verification_folder="$(mktemp -d "${TMPDIR:-/tmp}/cardiac-prism-release.XXXXXXXX")"
trap 'rm -rf -- "$verification_folder"' EXIT

if gh release view "$release_tag" --json isDraft --jq .isDraft > "$verification_folder/is-draft"; then
  is_draft="$(cat "$verification_folder/is-draft")"
else
  gh release create "$release_tag" --verify-tag --draft \
    --title "Cardiac-Prism $release_tag" --notes-file "$release_notes"
  is_draft=true
fi
if [[ "$is_draft" != true && "$is_draft" != false ]]; then
  echo 'Cannot determine whether the release is a draft.' >&2
  exit 1
fi
if [[ "$is_draft" == true ]]; then
  gh release upload "$release_tag" "$asset_folder"/* --clobber
fi
mkdir "$verification_folder/download"
gh release download "$release_tag" --dir "$verification_folder/download"
(
  cd "$verification_folder/download"
  cmp "$asset_folder/SHA256SUMS" SHA256SUMS
  sha256sum --strict --check "$asset_folder/SHA256SUMS"
)
if [[ "$is_draft" == true ]]; then
  gh release edit "$release_tag" --draft=false
fi
gh release view "$release_tag" --json url,assets --jq '{url, assets: [.assets[].name]}'
