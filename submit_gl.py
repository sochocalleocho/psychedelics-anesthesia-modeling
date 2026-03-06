#!/usr/bin/env python3
"""
Great Lakes GitOps Job Submission Agent Utility.
Run this locally to submit a Slurm script to Great Lakes via the GitOps workflow.

Usage:
    python submit_gl.py path/to/script.slurm
"""

import sys
import os
import shutil
import subprocess
from datetime import datetime

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR running command {cmd}:")
        print(e.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python submit_gl.py path/to/script.slurm")
        sys.exit(1)
        
    slurm_file = sys.argv[1]
    if not os.path.exists(slurm_file):
        print(f"Error: File {slurm_file} not found.")
        sys.exit(1)
        
    repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip()
    pending_dir = os.path.join(repo_root, 'gl_jobs', 'pending')
    
    # Ensure directory exists
    os.makedirs(pending_dir, exist_ok=True)
    
    # Copy file to pending directory
    basename = os.path.basename(slurm_file)
    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(basename)
    new_basename = f"{name}_{timestamp}{ext}"
    
    dest_path = os.path.join(pending_dir, new_basename)
    shutil.copy2(slurm_file, dest_path)
    
    print(f"Copied script to {dest_path}")
    
    # Git add, commit, push
    run_command(['git', 'add', dest_path], cwd=repo_root)
    run_command(['git', 'commit', '-m', f"GitOps(LocalAgent): Submitting {new_basename} to Great Lakes"], cwd=repo_root)
    print("Pushing to GitHub...")
    run_command(['git', 'push'], cwd=repo_root)
    
    print("\nSUCCESS!")
    print(f"The file {new_basename} has been sent to Great Lakes via GitOps.")
    print("The Great Lakes polling daemon should pick it up within 60 seconds.")
    print("Check git pull later to see the Job ID appear in gl_jobs/submitted/ and gl_jobs/logs/.")

if __name__ == "__main__":
    main()
