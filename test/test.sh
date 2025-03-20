#!/bin/bash

# Ensure the script stops if any command fails
set -e

echo "Running unit tests for EZ Diffusion Model..."

# Navigate to the project root (if necessary)
cd "$(dirname "$0")/.." || exit 1

# Run unittest to discover and execute all tests in the test/ directory
python3 -m unittest discover test

echo "Test Finished."

# ============================== 
# End of the new version code. 