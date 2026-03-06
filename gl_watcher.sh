#!/bin/bash
# Great Lakes GitOps Polling Daemon
# Run this inside a tmux session on Great Lakes
# USAGE: tmux new -s watcher ./gl_watcher.sh

# Directory config
REPO_DIR="/home/soichi/Psychedelics-Anesthesia-Modeling-Study"
JOBS_DIR="${REPO_DIR}/gl_jobs"
PENDING_DIR="${JOBS_DIR}/pending"
SUBMITTED_DIR="${JOBS_DIR}/submitted"
LOGS_DIR="${JOBS_DIR}/logs"

# Ensure directories exist
mkdir -p "${PENDING_DIR}" "${SUBMITTED_DIR}" "${LOGS_DIR}"

echo "Starting Great Lakes GitOps Watcher in ${REPO_DIR}"
echo "Polling every 60 seconds..."

while true; do
    cd "${REPO_DIR}" || { echo "ERROR: Could not cd to ${REPO_DIR}"; sleep 60; continue; }
    
    # 1. Pull latest changes (quietly)
    git pull --quiet origin main
    
    # 2. Check for pending jobs
    if [ "$(ls -A ${PENDING_DIR}/*.slurm 2>/dev/null)" ]; then
        for slurm_script in ${PENDING_DIR}/*.slurm; script_name=$(basename "${slurm_script}"); do
            echo "[$(date)] Found new job script: ${script_name}"
            
            # Submit the job
            submit_output=$(sbatch "${slurm_script}")
            
            if [ $? -eq 0 ]; then
                # Extract Job ID
                # Example sbatch output: "Submitted batch job 1234567"
                job_id=$(echo "${submit_output}" | awk '{print $NF}')
                echo "[$(date)] Successfully submitted job ${job_id}"
                
                # Move to submitted directory
                mv "${slurm_script}" "${SUBMITTED_DIR}/"
                
                # Create a log file with the submission info
                echo "${submit_output}" > "${LOGS_DIR}/${script_name}.job_id_${job_id}.log"
                
                # Commit and push back to GitHub
                git add "${SUBMITTED_DIR}/${script_name}" "${LOGS_DIR}/${script_name}.job_id_${job_id}.log" "${PENDING_DIR}/${script_name}"
                git commit -m "GitOps(GreatLakes): Submitted ${script_name} as Job ${job_id}"
                git push --quiet origin main || echo "[$(date)] WARNING: git push failed. Will retry later."
            else
                echo "[$(date)] ERROR: Failed to submit ${script_name}"
                echo "${submit_output}" > "${LOGS_DIR}/${script_name}.ERROR.log"
                git add "${LOGS_DIR}/${script_name}.ERROR.log"
                git commit -m "GitOps(GreatLakes): ERROR submitting ${script_name}"
                git push --quiet origin main || echo "[$(date)] WARNING: git push failed. Will retry later."
            fi
        done
    fi
    
    # Wait 60 seconds before next poll
    sleep 60
done
