import os
import streamlit.cli as stcli
import sys

def main():
    script_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
