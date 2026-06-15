"""Wrapper to run the EfficientNet evaluation script.
This simply invokes the existing `src/evaluate_efficientnet.py` module.
"""
import subprocess
import sys

def main():
    # Run the evaluation script using the same interpreter
    script_path = "src/evaluate_efficientnet.py"
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print("Evaluation completed with exit code", result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed (exit {e.returncode}):", e)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
