#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path

import hashlib

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_db_integrity(db_path):
    """
    Check if the db directory exists and contains necessary subdirectories.
    
    Args:
        db_path (Path): Path to the db directory.
        
    Returns:
        bool: True if db is complete, False otherwise.
    """
    db_path = Path(db_path)
    if not db_path.exists() or not db_path.is_dir():
        return False
        
    # Check for essential subdirectories
    required_subdirs = ["interproscan"]
    for subdir in required_subdirs:
        if not (db_path / subdir).exists():
            print(f"Missing required subdirectory: {subdir}")
            return False
            
    return True

import subprocess

def download_file(url, output_path):
    """
    Download a file from a URL to a local path using wget -c.
    
    Args:
        url (str): URL to download from.
        output_path (Path): Local path to save the file.
    """
    print(f"Downloading from {url}...")
    try:
        # Check if wget is installed
        if shutil.which("wget") is None:
            print("Error: wget is not installed.")
            return False
            
        cmd = ["wget", "-c", url, "-O", str(output_path)]
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("\nDownload complete.")
            return True
        else:
            print(f"\nwget exited with code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading file: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def extract_tar_gz(tar_path, extract_path):
    """
    Extract a tar.gz file.
    
    Args:
        tar_path (Path): Path to the tar.gz file.
        extract_path (Path): Directory to extract to.
    """
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def setup_db(project_root):
    """
    Main function to ensure db directory is ready.
    
    Args:
        project_root (Path): Root directory of the project.
        
    Returns:
        bool: True if db is ready, False otherwise.
    """
    project_root = Path(project_root)
    db_path = project_root / "db"
    tar_path = project_root / "db.tar.gz"
    
    # URL for db.tar.gz (Replace with actual URL if known, using a placeholder for now based on repo)
    # Assuming the file is released in the repo or hosted somewhere accessible.
    # Since I don't have the exact URL, I'll use a likely one or a placeholder.
    # User said "from githup download", so likely:
    # Keep the current release asset URL until a renamed iTAK Next release asset is published.
    DB_DOWNLOAD_URL = "https://github.com/Enver-Kastrioti/itak2_9.19/releases/download/iTAK2/db.tar.gz"
    EXPECTED_SHA256 = "a6121bff3a37096990fb2422fb7e7419db7b11423aa07e284a1f5c8b90926c5c"
    
    # 1. Check integrity
    if check_db_integrity(db_path):
        print("DB directory check passed.")
        return True
    
    print("DB directory missing or incomplete.")
    
    # 2. Check if tar.gz exists
    if not tar_path.exists():
        print(f"db.tar.gz not found at {tar_path}.")
        print("Attempting to download from GitHub...")
        if not download_file(DB_DOWNLOAD_URL, tar_path):
            print("Failed to download database.")
            return False
    else:
        print(f"Found db.tar.gz at {tar_path}.")
        
    # Verify SHA256 of the tarball (whether newly downloaded or existing)
    print("Verifying file integrity (SHA256)...")
    file_hash = calculate_sha256(tar_path)
    if file_hash != EXPECTED_SHA256:
        print(f"Error: SHA256 mismatch!")
        print(f"Expected: {EXPECTED_SHA256}")
        print(f"Actual:   {file_hash}")
        print("The file might be corrupted. Deleting it and aborting.")
        try:
            tar_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete corrupted file: {e}")
        return False
        
    print("SHA256 verification passed.")
        
    # 3. Extract
    if not extract_tar_gz(tar_path, project_root):
        return False
        
    # 4. Verify again
    if not check_db_integrity(db_path):
        print("DB integrity check failed after extraction.")
        return False
        
    # 5. Cleanup
    try:
        tar_path.unlink()
        print(f"Deleted {tar_path}.")
    except Exception as e:
        print(f"Warning: Could not delete {tar_path}: {e}")
        
    return True
