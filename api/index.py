import os
import sys
from pathlib import Path

# Add project root and backend dir to system path so backend imports resolve correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.app import create_app

# Create Flask app instance for Vercel Python runtime
app = create_app()
