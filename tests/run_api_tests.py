#!/usr/bin/env python
"""
Helper script to run tests with API keys loaded from .env file.
This script ensures that environment variables are properly loaded
before running the tests.

Usage:
    python tests/run_api_tests.py [pytest_args]

Example:
    python tests/run_api_tests.py -v tests/test_llm_manager.py::test_real_api_integration
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Get the project root directory
    repo_root = Path(__file__).parent.parent.absolute()
    
    # Load environment variables from .env file
    env_file = repo_root / '.env'
    if env_file.exists():
        print(f"Loading environment variables from {env_file}")
        load_dotenv(dotenv_path=env_file)
    else:
        print(f"ERROR: .env file not found at {env_file}")
        sys.exit(1)
    
    # Check for required API keys
    llm_api_key = os.getenv('LLM_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not llm_api_key:
        print("ERROR: LLM_API_KEY not found in .env file")
        sys.exit(1)
        
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    # Set USE_REAL_API=1 for testing
    os.environ['USE_REAL_API'] = '1'
    
    # Get pytest arguments
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else ['-v', 'tests/test_llm_manager.py::test_real_api_integration']
    
    # Print test info
    print(f"Running tests with real API keys")
    print(f"DeepSeek API Key: {'[SET]' if llm_api_key else '[NOT SET]'}")
    print(f"OpenAI API Key: {'[SET]' if openai_api_key else '[NOT SET]'}")
    print(f"USE_REAL_API: {os.environ['USE_REAL_API']}")
    print(f"Command: pytest {' '.join(pytest_args)}")
    
    # Run pytest with the environment variables
    env = os.environ.copy()
    cmd = ['python', '-m', 'pytest'] + pytest_args
    
    try:
        # Change to the repository root directory
        os.chdir(repo_root)
        # Run pytest in a subprocess
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 