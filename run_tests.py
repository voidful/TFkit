#!/usr/bin/env python3
"""Test runner script for TFKit."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_unit_tests(verbose=False, coverage=True):
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    cmd = ["python", "-m", "pytest", "tests/", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=tfkit",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("🔗 Running integration tests...")
    
    cmd = ["python", "-m", "pytest", "tests/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_all_tests(verbose=False, coverage=True, slow=False):
    """Run all tests."""
    print("🚀 Running all tests...")
    
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=tfkit",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_linting():
    """Run code linting."""
    print("🔍 Running code linting...")
    
    # Check if flake8 is available
    try:
        result = subprocess.run(["flake8", "tfkit/", "--max-line-length=100"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Linting passed!")
            return True
        else:
            print("❌ Linting failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠️  flake8 not found, skipping linting")
        print("   Install with: pip install flake8")
        return True


def run_type_checking():
    """Run type checking with mypy."""
    print("🔍 Running type checking...")
    
    try:
        result = subprocess.run(["mypy", "tfkit/", "--ignore-missing-imports"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Type checking passed!")
            return True
        else:
            print("❌ Type checking failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠️  mypy not found, skipping type checking")
        print("   Install with: pip install mypy")
        return True


def check_test_coverage():
    """Check test coverage and report."""
    if not Path("htmlcov").exists():
        print("⚠️  No coverage report found. Run tests with --coverage first.")
        return
    
    try:
        # Open coverage report
        coverage_file = Path("htmlcov/index.html")
        if coverage_file.exists():
            print(f"📊 Coverage report available at: {coverage_file.absolute()}")
        
        # Show coverage summary
        result = subprocess.run(["python", "-m", "coverage", "report"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 Coverage Summary:")
            print(result.stdout)
    except FileNotFoundError:
        print("⚠️  coverage not found")


def setup_test_environment():
    """Setup test environment."""
    print("🔧 Setting up test environment...")
    
    # Set environment variables for testing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Check if required packages are installed
    required_packages = ["pytest", "torch", "transformers"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ Test environment ready!")
    return True


def clean_test_artifacts():
    """Clean test artifacts."""
    print("🧹 Cleaning test artifacts...")
    
    artifacts = [
        ".pytest_cache",
        "htmlcov",
        ".coverage",
        "__pycache__",
        "*.pyc"
    ]
    
    for artifact in artifacts:
        path = Path(artifact)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()
    
    # Clean pycache directories recursively
    for pycache in Path(".").rglob("__pycache__"):
        import shutil
        shutil.rmtree(pycache)
    
    print("✅ Test artifacts cleaned!")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="TFKit Test Runner")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--type-check", action="store_true", help="Run type checking only")
    parser.add_argument("--coverage", action="store_true", default=True, help="Generate coverage report")
    parser.add_argument("--no-coverage", action="store_false", dest="coverage", help="Skip coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--clean", action="store_true", help="Clean test artifacts and exit")
    parser.add_argument("--setup", action="store_true", help="Setup test environment and exit")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.clean:
        clean_test_artifacts()
        return 0
    
    if args.setup:
        success = setup_test_environment()
        return 0 if success else 1
    
    # Setup environment
    if not setup_test_environment():
        return 1
    
    success = True
    
    # Run specific tests
    if args.unit:
        success &= run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success &= run_integration_tests(args.verbose)
    elif args.lint:
        success &= run_linting()
    elif args.type_check:
        success &= run_type_checking()
    else:
        # Run all tests and checks
        success &= run_linting()
        success &= run_type_checking()
        success &= run_all_tests(args.verbose, args.coverage, args.slow)
    
    # Show coverage report if generated
    if args.coverage and (not args.lint and not args.type_check):
        check_test_coverage()
    
    # Summary
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 