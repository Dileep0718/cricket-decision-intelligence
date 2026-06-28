import subprocess
import sys
import os

# Seed vector store on first run
from data.seed_matches import seed_vector_store
seed_vector_store()

# Run the app
from frontend.app import main
main()