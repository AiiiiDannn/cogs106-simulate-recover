#!/bin/bash
# src/main.sh
# This script runs the complete 3000-iteration simulate-and-recover experiment.

echo "Starting EZ Diffusion Model Simulate-and-Recover Experiment..."
python3 -m src.simulate_and_recover
echo "Experiment completed."