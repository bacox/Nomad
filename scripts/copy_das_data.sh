

# Copy the data using rsync from the DAS server to the local machine

#!/bin/bash
# Define the source and destination directories
SOURCE_DIR="das6_2:/var/scratch/bacox/mobile-async-fl/results"
DEST_DIR="/home/bacox/Documents/Bart/mobile-async-fl/das_results"
# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"
# Use rsync to copy the data from the source to the destination
rsync -avz --progress "$SOURCE_DIR/" "$DEST_DIR/"
# Print a message indicating that the copy is complete
echo "Data copy from DAS server to local machine completed successfully."