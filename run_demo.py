import os
import sys
import time
import socket
import threading
import webbrowser

def find_available_port(preferred_port=5000):
    """Returns preferred_port if available, otherwise finds an open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', preferred_port)) != 0:
            return preferred_port
    # Find next open port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)

    print("========================================================================")
    print("  FLOOD & WATERLOGGING SEGMENTATION & ROAD PASSABILITY ESTIMATION")
    print("  Capstone Project Demo Web Application")
    print("  Ramdeobaba University, Nagpur | Dept of Electronics Engineering")
    print("========================================================================")

    # Add paths
    for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src'), os.path.join(PROJECT_ROOT, 'web_app')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import Flask app and load models
    try:
        from web_app.app import app, load_models
    except ImportError:
        # pyrefly: ignore[missing-import]
        from app import app, load_models

    load_models()

    port = find_available_port(5000)
    url = f"http://127.0.0.1:{port}"

    print(f"\nStarting Flask Web Server...")
    print(f"Dashboard available at: {url}\n")
    print("Press Ctrl+C to stop the server.\n")

    # Open browser in a background thread after server starts listening
    def _open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n[Shutting down web server cleanly...]")
    except Exception as e:
        print(f"\n[Flask Error]: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
