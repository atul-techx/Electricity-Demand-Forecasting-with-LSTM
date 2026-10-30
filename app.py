"""
app.py (Workspace Root Entrypoint)
----------------------------------
Entrypoint so running:
    uvicorn app:app --reload
or:
    python app.py
from the workspace root runs the FastAPI application seamlessly.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "electricity_forecasting")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Import directly from the electricity_forecasting app module
from electricity_forecasting.app import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  EDF-LSTM  |  FastAPI Application Starting (Workspace Root)")
    print("  Open dashboard at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
