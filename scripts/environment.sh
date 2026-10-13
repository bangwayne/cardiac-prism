#!/usr/bin/env bash
# Accept configuration from earlier workstation installations.
# Explicit CARDIAC_PRISM_* settings take precedence over legacy aliases.
cardiac_prism_legacy_environment() {
  local legacy_name current_name option
  for legacy_name in ${!KPRISM_@}; do
    current_name="CARDIAC_PRISM_${legacy_name#KPRISM_}"
    if [[ ! -v "$current_name" ]]; then
      printf -v "$current_name" '%s' "${!legacy_name}"
      export "$current_name"
    fi
  done
  for option in CONFIG CHECKPOINT DEVICE; do
    legacy_name="VERSE_$option"
    current_name="CARDIAC_PRISM_$option"
    if [[ -v "$legacy_name" && ! -v "$current_name" ]]; then
      printf -v "$current_name" '%s' "${!legacy_name}"
      export "$current_name"
    fi
  done
}
cardiac_prism_legacy_environment
unset -f cardiac_prism_legacy_environment
