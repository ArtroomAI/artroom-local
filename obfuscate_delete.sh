#!/bin/bash

startDir=$(pwd)

# Find and delete all .pyc files recursively
find "$startDir" -type f -name "*.pyc" -print0 | while IFS= read -r -d $'\0' file; do
    echo "Deleting: $file"
    rm -f "$file"
done

# Add 'pause' behavior by waiting for user input
read -rp "Press Enter to exit..."
