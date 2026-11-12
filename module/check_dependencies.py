#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAK Next dependency checking module.

Checks whether the required Python packages and external tools are available.
"""

import sys
import subprocess
import importlib
import shutil
from pathlib import Path
import os
import platform
import tarfile

try:
    from module.runtime_tools import (
        activate_bundled_interproscan_binaries,
        resolve_helper_executable,
        resolve_java_executable,
    )
except ImportError:
    activate_bundled_interproscan_binaries = None
    resolve_helper_executable = None
    resolve_java_executable = None

class DependencyChecker:
    """Dependency checker."""
    
    def __init__(self, interproscan_path=None):
        self.missing_dependencies = []
        self.missing_optional_dependencies = []
        self.warnings = []
        self.interproscan_path = Path(interproscan_path) if interproscan_path else None
        
        # Required Python packages (needed for core functionality)
        self.required_python_packages = {
            'Bio': 'Biopython (bioinformatics library)',
            'pandas': 'pandas (data analysis library)',
            'numpy': 'NumPy (numerical computing library)',
            'json': 'json (standard library)',
            'csv': 'csv (standard library)',
            'argparse': 'argparse (standard library)',
            'subprocess': 'subprocess (standard library)',
            'pathlib': 'pathlib (standard library)',
            'os': 'os (standard library)',
            'sys': 'sys (standard library)',
            'time': 'time (standard library)',
            'datetime': 'datetime (standard library)',
            'shutil': 'shutil (standard library)',
            'collections': 'collections (standard library)',
            'warnings': 'warnings (standard library)'
        }
        
        # Optional Python packages (needed for specific features)
        self.optional_python_packages = {
            'torch': 'PyTorch (required only for prediction)',
            'matplotlib': 'matplotlib (required by prediction / Grad-CAM plotting)'
        }
        
        # Required external tools
        # hmmscan is preferentially resolved from the bundled InterProScan distribution when present;
        # for compatibility we keep checks, but handle hmmscan specially.
        self.required_external_tools = {
            'java': 'Java runtime (required by InterProScan)',
            'perl': 'Perl interpreter (required by InterProScan)',
            'python3': 'Python 3 interpreter'
        }
        
        # hmmscan is handled separately because it may be bundled
        self.hmmscan_tool_name = 'hmmscan'
        
        # Key paths (module lives under module/, so project root is one level up)
        self.script_dir = Path(__file__).parent.parent.absolute()
        self.db_dir = self.script_dir / "db"
        self.db_archive = self.script_dir / "db.tar.gz"
        
        self.required_files = {
            'interproscan.sh': self.interproscan_path if self.interproscan_path else self.script_dir / "db" / "interproscan" / "interproscan.sh",
            'self_build.hmm': self.script_dir / "hmm" / "self_build.hmm",
            'predict.py': self.script_dir / "pre_model" / "predict.py",
            'model.pth': self.script_dir / "pre_model" / "model.pth"
        }
    
    def check_python_package(self, package_name):
        """Check whether a Python package is installed."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def check_external_tool(self, tool_name):
        return shutil.which(tool_name) is not None

    def check_java_runtime(self):
        """Validate that Java is invocable and meets the InterProScan minimum version."""
        java_candidate = resolve_java_executable(self.script_dir) if resolve_java_executable else None
        java_path = str(java_candidate) if java_candidate else shutil.which("java")
        if not java_path:
            return False, "Java runtime (required by InterProScan) was not found on PATH"

        try:
            result = subprocess.run(
                [java_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
        except Exception as e:
            return False, f"Java was found at {java_path}, but could not be executed: {e}"

        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            if stderr:
                return False, f"Java was found at {java_path}, but 'java -version' failed: {stderr}"
            return False, f"Java was found at {java_path}, but 'java -version' failed"

        version_output = (result.stderr or result.stdout or "").strip()
        version_line = ""
        for line in version_output.splitlines():
            if "version" in line.lower():
                version_line = line.strip()
                break

        if not version_line:
            return False, f"Java was found at {java_path}, but its version could not be determined"

        version_str = ""
        if '"' in version_line:
            try:
                version_str = version_line.split('"')[1]
            except Exception:
                version_str = ""

        major_version = None
        if version_str:
            try:
                if version_str.startswith("1."):
                    major_version = int(version_str.split(".")[1])
                else:
                    major_version = int(version_str.split(".")[0])
            except Exception:
                major_version = None

        if major_version is None:
            return False, f"Java was found at {java_path}, but its version could not be parsed: {version_line}"

        if major_version < 11:
            return False, f"Java {version_str} found at {java_path}, but InterProScan requires Java 11+"

        return True, f"Java {version_str} available at {java_path}"

    def check_hmmscan(self):
        """Check whether hmmscan is available (bundled or system)."""
        helper_hmmscan = None
        if resolve_helper_executable is not None:
            helper_hmmscan = resolve_helper_executable(self.script_dir, "hmmer3", "hmmscan")
        if helper_hmmscan is not None:
            return True, f"Using platform helper hmmscan: {helper_hmmscan}"

        # 0) If an external InterProScan path is provided, prioritize its bundled hmmscan
        if self.interproscan_path:
            interpro_dir = self.interproscan_path.parent
            hmmscan_candidates = []
            # Common path: bin/hmmer/hmmer3/hmmscan
            hmmscan_candidates.append(interpro_dir / "bin" / "hmmer" / "hmmer3" / "hmmscan")
            # Common variant: bin/hmmer/hmmscan
            hmmscan_candidates.append(interpro_dir / "bin" / "hmmer" / "hmmscan")
            # Common variant: bin/hmmscan
            hmmscan_candidates.append(interpro_dir / "bin" / "hmmscan")
            # Search within hmmer3 directory (fallback)
            hmmer3_dir = interpro_dir / "bin" / "hmmer" / "hmmer3"
            if hmmer3_dir.exists():
                for path in hmmer3_dir.rglob("hmmscan"):
                    hmmscan_candidates.append(path)
            # Search within bin directory (final fallback)
            bin_dir = interpro_dir / "bin"
            if bin_dir.exists():
                for path in bin_dir.rglob("hmmscan"):
                    hmmscan_candidates.append(path)
            for cand in hmmscan_candidates:
                try:
                    if cand.exists() and os.access(cand, os.X_OK) and cand.is_file():
                        return True, f"Using bundled hmmscan from custom InterProScan: {cand}"
                except Exception:
                    pass
        
        # 1) Check bundled hmmscan
        internal_hmmscan = self.db_dir / "interproscan" / "bin" / "hmmer" / "hmmer3" / "hmmscan"
        if internal_hmmscan.exists() and os.access(internal_hmmscan, os.X_OK):
            return True, f"Using bundled hmmscan: {internal_hmmscan}"
            
        # 2) Check system hmmscan
        if shutil.which("hmmscan"):
            return True, "Using system hmmscan"
            
        return False, "hmmscan was not found (neither bundled nor on system PATH)"
    
    def check_file_exists(self, file_path):

        return file_path.exists()

    def _detect_binary_format(self, binary_path):
        """Return a lightweight binary format label for the given file."""
        try:
            with open(binary_path, "rb") as f:
                header = f.read(4)
        except Exception:
            return "unknown"

        if header.startswith(b"#!"):
            return "script"
        if header == b"\x7fELF":
            return "elf"
        if header in (
            b"\xfe\xed\xfa\xce",
            b"\xfe\xed\xfa\xcf",
            b"\xce\xfa\xed\xfe",
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xd0\x0d",
            b"\x0d\xd0\xfe\xca",
        ):
            return "mach-o"
        return "unknown"

    def check_interproscan_native_binaries(self, interproscan_dir):
        """Validate that bundled native binaries match the current platform."""
        interproscan_dir = Path(interproscan_dir)
        current_system = platform.system().lower()

        candidates = [
            interproscan_dir / "bin" / "hmmer" / "hmmer3" / "3.3" / "hmmsearch",
            interproscan_dir / "bin" / "cdd" / "rpsblast",
            interproscan_dir / "bin" / "prosite" / "pfscanV3",
        ]

        for candidate in candidates:
            if not candidate.exists():
                continue

            fmt = self._detect_binary_format(candidate)
            if fmt == "unknown":
                continue

            if current_system == "darwin" and fmt == "elf":
                return False, f"InterProScan binary is Linux-only on this macOS host: {candidate}"
            if current_system == "linux" and fmt == "mach-o":
                return False, f"InterProScan binary is macOS-only on this Linux host: {candidate}"

        return True, "InterProScan native binaries look compatible with the current platform"
    
    @staticmethod
    def _find_interproscan_bin_dirs(interproscan_dir):
        """Return a list of InterProScan binary subdirectories that may need executable permissions."""
        interproscan_dir = Path(interproscan_dir)
        candidates = [
            interproscan_dir / "bin",
        ]
        result = []
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                result.append(str(candidate))
        return result

    def check_interproscan_setup(self):
        # If an external InterProScan path is specified, validate that installation
        if self.interproscan_path:
            if not self.interproscan_path.exists():
                return False
                
            if not os.access(self.interproscan_path, os.X_OK):
                self.warnings.append(f"Specified InterProScan script is not executable: {self.interproscan_path}")
                return False
                
            # Check InterProScan data directory
            interproscan_dir = self.interproscan_path.parent
            data_dir = interproscan_dir / "data"
            
            if not data_dir.exists():
                self.warnings.append(f"InterProScan data directory does not exist: {data_dir}")
                return False

            if (
                activate_bundled_interproscan_binaries is not None
                and interproscan_dir.resolve() == (self.script_dir / "db" / "interproscan").resolve()
            ):
                activated, activation_msg = activate_bundled_interproscan_binaries(self.script_dir, interproscan_dir)
                if not activated:
                    self.warnings.append(activation_msg)
                    return False

            native_ok, native_msg = self.check_interproscan_native_binaries(interproscan_dir)
            if not native_ok:
                self.warnings.append(native_msg)
                return False
                
            return True

        interproscan_script = self.required_files['interproscan.sh']
        
        if not interproscan_script.exists():
            return False
        
        # Check that the InterProScan script is executable
        if not os.access(interproscan_script, os.X_OK):
            self.warnings.append(f"InterProScan script is not executable: {interproscan_script}")
            return False
        
        # Check InterProScan data directory
        interproscan_dir = interproscan_script.parent
        data_dir = interproscan_dir / "data"
        
        if not data_dir.exists():
            self.warnings.append(f"InterProScan data directory does not exist: {data_dir}")
            return False

        if activate_bundled_interproscan_binaries is not None:
            activated, activation_msg = activate_bundled_interproscan_binaries(self.script_dir, interproscan_dir)
            if not activated:
                self.warnings.append(activation_msg)
                return False

        # Auto-fix executable permissions on InterProScan bundled binaries
        bin_paths = self._find_interproscan_bin_dirs(interproscan_dir)
        total_fixed = 0
        for bdir in bin_paths:
            for root, dirs, files in os.walk(bdir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if not os.access(fpath, os.X_OK):
                        try:
                            st = os.stat(fpath)
                            # Only add x if it's already readable (not a permission-denied file)
                            if st.st_mode & 0o444:
                                os.chmod(fpath, st.st_mode | 0o111)
                                total_fixed += 1
                        except Exception:
                            pass
        if total_fixed > 0:
            self.warnings.append(f'Auto-fixed {total_fixed} InterProScan binary permissions (chmod +x)')

        native_ok, native_msg = self.check_interproscan_native_binaries(interproscan_dir)
        if not native_ok:
            self.warnings.append(native_msg)
            return False
        
        return True
    
    def check_pytorch_gpu_support(self):
        """Check whether PyTorch has GPU support (may require importing torch)."""
        try:
            # We avoid heavy CUDA initialization unless the user explicitly requests GPU diagnostics.
            # Here we only attempt detailed checks when torch is installed.
            
            if importlib.util.find_spec('torch') is None:
                return "PyTorch is not installed"

            # Note: importing torch may still be slow; this function is called only at the end of run_full_check.
            
            import torch
            
            # Check PyTorch version
            torch_version = torch.__version__
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                return f"GPU support available - PyTorch {torch_version} (CUDA {cuda_version}), device count: {gpu_count}, primary device: {gpu_name}"
            else:
                # Check whether this is a CPU-only build
                if '+cpu' in torch_version:
                    return f"PyTorch {torch_version} (CPU-only build) - install a CUDA build for GPU acceleration"
                else:
                    return f"PyTorch {torch_version} - GPU support not available; computations will run on CPU"
        except ImportError:
            return "PyTorch is not installed; GPU support cannot be evaluated"

    def check_biopython_features(self):
        try:
            import Bio
            from Bio.Seq import Seq
            ver = getattr(Bio, "__version__", "unknown")
            _ = Seq("ATG").translate(table=1)
            _ = Seq("ATG").reverse_complement()
            return True, f"Biopython version: {ver}; Seq.translate/reverse_complement are available"
        except Exception as e:
            return False, f"Biopython feature check failed: {e}"
    
    def check_prediction_dependencies(self):

        for package in ('torch', 'matplotlib'):
            if not self.check_python_package(package):
                return False, f"Prediction requires {package}, but {package} is not installed"
        
        # Check prediction model file
        model_file = self.required_files.get('model.pth')
        if not self.check_file_exists(model_file):
            return False, f"Prediction requires a model file, but it does not exist: {model_file}"
        
        predict_script = self.required_files.get('predict.py')
        if not self.check_file_exists(predict_script):
            return False, f"Prediction requires a prediction script, but it does not exist: {predict_script}"
        
        return True, "Prediction dependencies satisfied"
    
    def ensure_db_extracted(self):
        """
        Ensure that the db directory is available.

        If the db directory is missing, attempt to extract it from an archive or download it.
        When an external InterProScan is provided, db/interproscan is not strictly required, but
        other assets (e.g., HMM files) may still be needed.

        Implemented via module/db_manager.py.
        
        Returns:
            bool: True if the db directory is usable; otherwise False.
        """
        try:
            # Dynamically import db_manager (located in the same module directory)
            
            # Resolve db_manager.py path
            current_dir = Path(__file__).parent.absolute()
            db_manager_path = current_dir / "db_manager.py"
            
            if not db_manager_path.exists():
                print(f"  [ERROR] db_manager module not found: {db_manager_path}")
                # Fallback: only check existence
                if self.db_dir.exists() and self.db_dir.is_dir():
                    return True
                return False
                
            import importlib.util
            spec = importlib.util.spec_from_file_location("db_manager", db_manager_path)
            db_manager = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db_manager)
            
            # Call setup_db using the project root directory
            return db_manager.setup_db(self.script_dir)
                
        except Exception as e:
            print(f"  [ERROR] Error while checking/preparing db directory: {e}")
            return False

    def run_full_check(self, check_predict=False, skip_db_check=False, test_mode=False):
        """
        Run the full dependency check.
        
        Args:
            check_predict (bool): Whether to check prediction dependencies (e.g., PyTorch).
            skip_db_check (bool): Whether to skip db directory preparation/checks.
            test_mode (bool): Whether running in test mode (skip InterProScan/hmmscan checks).
        """
        print("Starting iTAK Next dependency checks...")
        print("=" * 60)
        
        all_dependencies_met = True
        
        # Ensure db directory is ready (unless skipped)
        # If the user provides --interproscan, the local db directory is not required.
        if test_mode:
            print("\nDatabase files:")
            print("  [SKIP] Test mode (-test) does not require the db directory")
        elif skip_db_check:
            print("\nDatabase files:")
            print("  [SKIP] db checks are disabled")
        elif self.interproscan_path:
            print("\nDatabase files:")
            print("  [SKIP] --interproscan specified; skipping db directory preparation checks")
        else:
            print("\nDatabase files:")
            if self.ensure_db_extracted():
                print("  [OK] Database assets are ready")
            else:
                print("  [ERROR] Database asset preparation failed")
                self.missing_dependencies.append("Database files")
                all_dependencies_met = False
        
        # Check required Python packages
        print("\nRequired Python packages:")
        for package, description in self.required_python_packages.items():
            if self.check_python_package(package):
                print(f"  [OK] {package:<15} - {description}")
            else:
                print(f"  [ERROR] {package:<15} - {description}")
                self.missing_dependencies.append(f"Python package: {package}")
                all_dependencies_met = False
        
        # Check optional Python packages used by prediction-related paths
        print("\nOptional Python packages:")
        for package, description in self.optional_python_packages.items():
            if not check_predict:
                continue

            if package == 'torch':
                print("  [INFO] Checking PyTorch (this may take a few seconds)...")
            
            if self.check_python_package(package):
                print(f"  [OK] {package:<15} - {description}")
            else:
                print(f"  [WARN] {package:<15} - {description}")
                self.missing_optional_dependencies.append(f"Python package: {package}")
                print(f"      Note: missing {package} disables prediction")

        # Additional check: Biopython core features
        print("\nBiopython feature checks:")
        ok, msg = self.check_biopython_features()
        if ok:
            print(f"  [OK] {msg}")
        else:
            print(f"  [ERROR] {msg}")
            self.missing_dependencies.append("Biopython core features")
            all_dependencies_met = False
        
        # Check external tools
        print("\nExternal tools:")
        
        if not test_mode:
            hmm_ok, hmm_msg = self.check_hmmscan()
            if hmm_ok:
                print(f"  [OK] hmmscan         - {hmm_msg}")
            else:
                print(f"  [ERROR] hmmscan         - {hmm_msg}")
                self.missing_dependencies.append("External tool: hmmscan")
                all_dependencies_met = False
            
        tools_to_check = self.required_external_tools.items()
        if test_mode:
            tools_to_check = [('python3', self.required_external_tools['python3'])]

        for tool, description in tools_to_check:
            if tool == 'java':
                ok, msg = self.check_java_runtime()
                if ok:
                    print(f"  [OK] {tool:<15} - {msg}")
                else:
                    print(f"  [ERROR] {tool:<15} - {msg}")
                    self.missing_dependencies.append(f"External tool: {tool}")
                    all_dependencies_met = False
                continue

            if self.check_external_tool(tool):
                print(f"  [OK] {tool:<15} - {description}")
            else:
                print(f"  [ERROR] {tool:<15} - {description}")
                self.missing_dependencies.append(f"External tool: {tool}")
                all_dependencies_met = False
        
        # Check required files
        print("\nKey files:")
        for file_name, file_path in self.required_files.items():
            # If prediction is not requested, skip prediction-related files
            if not check_predict and file_name in ['predict.py', 'model.pth']:
                continue
            if test_mode and file_name in ['interproscan.sh', 'self_build.hmm']:
                continue
                
            if self.check_file_exists(file_path):
                print(f"  [OK] {file_name:<20} - {file_path}")
            else:
                print(f"  [ERROR] {file_name:<20} - {file_path}")
                self.missing_dependencies.append(f"File: {file_path}")
                all_dependencies_met = False
        
        # Check InterProScan configuration
        if test_mode:
            print("\nInterProScan configuration:")
            print("  [SKIP] Test mode (-test) does not require InterProScan")
        else:
            print("\nInterProScan configuration:")
            if self.check_interproscan_setup():
                print("  [OK] InterProScan configuration is valid")
            else:
                print("  [ERROR] InterProScan configuration is invalid")
                self.missing_dependencies.append("InterProScan configuration")
                all_dependencies_met = False
        
        # Check PyTorch GPU support (only when prediction checks are enabled)
        if check_predict:
            print("\nGPU support (prediction-related):")
            if self.check_python_package('torch'):
                gpu_status = self.check_pytorch_gpu_support()
                print(f"  [INFO] {gpu_status}")
            else:
                print("  [WARN] PyTorch is not installed; GPU status cannot be evaluated")
        
        # Emit warnings
        if self.warnings:
            print("\n[WARN] Warnings:")
            for warning in self.warnings:
                print(f"  [WARN] {warning}")
        
        # Summary
        print("\n" + "=" * 60)
        if all_dependencies_met:
            print("[OK] All required dependencies are satisfied. iTAK Next is ready to run.")
            if self.missing_optional_dependencies:
                print("[WARN] The following optional dependencies are missing; some features may be unavailable:")
                for dep in self.missing_optional_dependencies:
                    print(f"  - {dep}")
                if check_predict:
                    print("  Install PyTorch to enable prediction.")
        else:
            print("[ERROR] Missing required dependencies. Please install the following components:")
            for dep in self.missing_dependencies:
                print(f"  - {dep}")
            print("\nInstallation suggestions:")
            self._print_installation_suggestions()
        
        return all_dependencies_met
    
    def _print_installation_suggestions(self):
        """Print installation suggestions."""
        print("\nInstallation suggestions:")
        
        # Python package suggestions
        python_packages_missing = [dep.split(': ')[1] for dep in self.missing_dependencies if dep.startswith('Python package: ')]
        if python_packages_missing:
            print("\n  Python packages:")
            for pkg in python_packages_missing:
                if pkg == 'torch':
                    print("    PyTorch (choose a build appropriate for your system):")
                    print("      CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
                    print("      GPU (CUDA): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                    print("      More options: https://pytorch.org/get-started/locally/")
                elif pkg == 'matplotlib':
                    print("    pip install matplotlib")
                elif pkg == 'Bio':
                    print(f"    pip install biopython")
                elif pkg in ['pandas', 'numpy']:
                    print(f"    pip install {pkg}")
        
        # External tool suggestions
        external_tools_missing = [dep.split(': ')[1] for dep in self.missing_dependencies if dep.startswith('External tool: ')]
        if external_tools_missing:
            print("\n  External tools:")
            for tool in external_tools_missing:
                if tool == 'hmmscan':
                    print("    Install HMMER: http://hmmer.org/download.html")
                    print("    Ubuntu/Debian: sudo apt-get install hmmer")
                    print("    CentOS/RHEL: sudo yum install hmmer")
                elif tool == 'java':
                    print("    Install Java: https://www.oracle.com/java/technologies/downloads/")
                    print("    Ubuntu/Debian: sudo apt-get install openjdk-11-jdk")
                    print("    CentOS/RHEL: sudo yum install java-11-openjdk")
                elif tool == 'perl':
                    print("    Install Perl: https://www.perl.org/get.html")
                    print("    Ubuntu/Debian: sudo apt-get install perl")
                    print("    CentOS/RHEL: sudo yum install perl")

def main():
    """Entry point."""
    checker = DependencyChecker()
    success = checker.run_full_check()
    
    if not success:
        sys.exit(1)
    
    return success

if __name__ == "__main__":
    main()
