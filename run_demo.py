import os
import sys
import webbrowser
import time

def main():
    print("========================================================================")
    print("  FLOOD & WATERLOGGING SEGMENTATION & ROAD PASSABILITY ESTIMATION")
    print("  Capstone Project Demo Web Application")
    print("  Ramdeobaba University, Nagpur | Dept of Electronics Engineering")
    print("========================================================================")
    
    app_path = os.path.join(os.path.dirname(__file__), 'web_app', 'app.py')
    print(f"\nStarting Flask Web Server from '{app_path}'...")
    print("Dashboard will be available at: http://127.0.0.1:5000\n")

    # Add sys path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_app')))

    from app import app, load_models
    load_models()

    try:
        webbrowser.open('http://127.0.0.1:5000')
    except Exception as e:
        print(f"Could not auto-open browser: {e}")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
