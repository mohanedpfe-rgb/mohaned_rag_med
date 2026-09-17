"""
Simple launcher for the deterministic RAG system UI.
This script launches the Streamlit interface with the new deterministic answer generation.
"""
import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the deterministic RAG system UI."""
    print("Launching Deterministic RAG System with Advanced Answer Generation")
    print("=" * 70)
    print("System Features:")
    print("  - 100% LLM-free answer generation")
    print("  - Advanced NLP sentence extraction")
    print("  - Multi-signal evidence ranking")
    print("  - Context-aware template generation")
    print("  - Comprehensive quality assessment")
    print("  - Multi-document synthesis")
    print("=" * 70)
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"Working directory: {project_dir}")
    print("Starting Streamlit server...")
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error launching server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import os
    sys.exit(main())