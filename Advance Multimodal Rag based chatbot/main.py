"""
main.py — Safe launcher for the Streamlit UI

Why this file exists:
- Many environments (and some deployment setups) expect a Python entrypoint.
- Streamlit apps are normally started with:  streamlit run ui/streamlit_app.py
- This launcher provides a single command:  python main.py
  which internally invokes the correct Streamlit command.

Important:
- This does not replace 'streamlit run'; it simply shells out to it.
- If you prefer, you can still run:  streamlit run ui/streamlit_app.py

to run: python -m streamlit run ui\chatui.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    # Resolve path to the Streamlit app
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    if not app_path.exists():
        print(f"[ERROR] Streamlit app not found at: {app_path}")
        print("Make sure your project structure includes ui/streamlit_app.py")
        return 1

    # Build the command: python -m streamlit run ui/streamlit_app.py
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]

    # Optional: pass through a custom port from env (STREAMLIT_SERVER_PORT)
    port = os.getenv("STREAMLIT_SERVER_PORT")
    if port:
        cmd += ["--server.port", port]

    # Optional: bind to all interfaces (useful in containers)
    cmd += ["--server.address", "0.0.0.0"]

    print("[INFO] Launching Streamlit app…")
    print("       Command:", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[ERROR] Streamlit is not installed or not on PATH.")
        print("Install with: pip install streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
