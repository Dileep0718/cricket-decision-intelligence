import subprocess
import sys
import os

sys.path.insert(0,os.path.dirname(__file__))

# Seed vector store on first run
try:
    from data.seed_matches import seed_vector_store
    seed_vector_store()
except Exception as e:
    print(f"vector store seeding warning : {e}")
# Run the app
from frontend.app import main
main()