# InterProScan Helper Binaries

This directory previously contained platform-specific helper binaries
copied from the bundled InterProScan distribution.

## Status

- **`bin/linux/`**: Linux helper binaries (kept in the repository for systems
  where the bundled InterProScan platform binaries are not on PATH).
- **`bin/osx/`**: macOS helper binaries **removed from the repository**
  (see Issue #7). macOS users should rely on system-installed tools
  (`brew install hmmer` etc.) or use `install_runtime.sh` to prepare
  helpers from the bundled InterProScan.

## Resolution priority (hmmscan example)

1. `bin/<platform>/hmmer3/hmmscan` — platform helper binary
2. `db/interproscan/bin/hmmer/hmmer3/hmmscan` — bundled InterProScan binary
3. System PATH (e.g., `/usr/bin/hmmscan`)
