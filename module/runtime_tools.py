#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import shutil
from pathlib import Path


INTERPROSCAN_BINARY_LINKS = {
    "cdd/rpsblast": "cdd/rpsblast",
    "cdd/rpsbproc": "cdd/rpsbproc",
    "hmmer/hmmer2/2.3.2/hmmpfam": "hmmer2/hmmpfam",
    "hmmer/hmmer3/3.3/hmmsearch": "hmmer3/hmmsearch",
    "hmmer/hmmer3/3.3/hmmscan": "hmmer3/hmmscan",
    "hmmer/hmmer3/3.3/hmmpress": "hmmer3/hmmpress",
    "hmmer/hmmer3/3.3/esl-translate": "hmmer3/esl-translate",
    "panther/epa-ng": "panther/epa-ng",
    "prosite/pfscanV3": "prosite/pfscanV3",
    "prosite/pfsearchV3": "prosite/pfsearchV3",
}


def get_platform_bin_name():
    current_system = platform.system().lower()
    if current_system == "darwin":
        return "osx"
    if current_system == "linux":
        return "linux"
    return None


def get_helper_root(script_dir):
    platform_bin = get_platform_bin_name()
    if not platform_bin:
        return None

    helper_root = Path(script_dir) / "bin" / platform_bin
    if helper_root.exists() and helper_root.is_dir():
        return helper_root
    # Platform-specific helpers are no longer bundled; fall through gracefully.
    return None


def resolve_helper_executable(script_dir, category, executable_name):
    helper_root = get_helper_root(script_dir)
    if helper_root is None:
        return None

    candidate = helper_root / category / executable_name
    if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def resolve_java_executable(script_dir):
    local_jdk_root = Path(script_dir) / ".local-jdk"
    if local_jdk_root.exists():
        candidates = list(local_jdk_root.glob("**/Contents/Home/bin/java"))
        candidates.extend(local_jdk_root.glob("**/bin/java"))
        for candidate in sorted(set(candidates)):
            if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    system_java = shutil.which("java")
    if system_java:
        return Path(system_java)
    return None


def build_runtime_env(script_dir):
    env = os.environ.copy()

    java_executable = resolve_java_executable(script_dir)
    if java_executable is not None:
        java_bin_dir = str(java_executable.parent)
        env["PATH"] = java_bin_dir + os.pathsep + env.get("PATH", "")
        java_home = java_executable.parent.parent
        if java_home.name == "Home" and java_home.parent.name == "Contents":
            java_home = java_home
        env.setdefault("JAVA_HOME", str(java_home))

    helper_root = get_helper_root(script_dir)
    if helper_root is not None:
        helper_lib = helper_root / "lib"
        if helper_lib.exists():
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = (
                str(helper_lib) if not existing else str(helper_lib) + os.pathsep + existing
            )

    return env


def activate_bundled_interproscan_binaries(script_dir, interproscan_dir=None):
    if platform.system().lower() != "darwin":
        return True, "No InterProScan binary activation required on this platform"

    helper_root = get_helper_root(script_dir)
    if helper_root is None:
        # bin/osx/ is no longer bundled in the repository.
        # On macOS, users should rely on system-installed tools or
        # run install_runtime.sh to prepare helper binaries.
        return True, "macOS helper binaries not bundled (use system tools or install_runtime.sh)"

    if interproscan_dir is None:
        interproscan_dir = Path(script_dir) / "db" / "interproscan"
    else:
        interproscan_dir = Path(interproscan_dir)

    interproscan_bin_dir = interproscan_dir / "bin"
    if not interproscan_bin_dir.exists():
        return False, f"InterProScan bin directory not found: {interproscan_bin_dir}"

    updated = 0
    missing = []

    for target_rel, source_rel in INTERPROSCAN_BINARY_LINKS.items():
        source = helper_root / source_rel
        target = interproscan_bin_dir / target_rel

        if not source.exists():
            missing.append(str(source))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        desired_link = os.path.relpath(source, target.parent)

        if target.is_symlink():
            current_link = os.readlink(target)
            if current_link == desired_link:
                continue
            target.unlink()
        elif target.exists():
            backup = target.with_name(target.name + ".bundled-orig")
            if not backup.exists():
                shutil.move(str(target), str(backup))
            else:
                target.unlink()

        os.symlink(desired_link, target)
        updated += 1

    if missing:
        return False, "Missing helper binaries: " + ", ".join(missing)

    return True, f"Activated {updated} macOS helper binaries for bundled InterProScan"
