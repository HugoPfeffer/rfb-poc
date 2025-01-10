import unittest
import sys
import time
import inspect
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import tempfile
import os
import pandas as pd
import numpy as np
from termcolor import colored

from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger
from src.validation.validation_framework import ValidationError
from src.generators.components import (
    DataComponent,
    DataPersistence,
    DataTransformation,
    FraudScenarioGenerator
)
from tests.test_generators.test_dataset_generator import TestDatasetGeneratorClass
from tests.test_generators.test_employment_generator import TestEmploymentGenerator
from tests.test_generators.test_investment_generator import TestInvestmentGenerator

class TestSection:
    """Represents a section of tests."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.total = 0
        self.failures: List[Tuple[unittest.TestCase, str]] = []
        self.start_time = 0.0
        self.end_time = 0.0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time for the section."""
        return self.end_time - self.start_time

class ColoredTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""
    
    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize with custom stream handler for colored output."""
        super().__init__(*args, **kwargs)
        self.stream = ColoredOutputWrapper(self.stream)
        self.start_time = None
        self.sections: Dict[str, TestSection] = {
            'TestDatasetGenerator': TestSection(
                'Base Dataset Generator Tests',
                'Tests for the base dataset generation functionality'
            ),
            'TestEmploymentGenerator': TestSection(
                'Employment Generator Tests',
                'Tests for employment data generation functionality'
            ),
            'TestInvestmentGenerator': TestSection(
                'Investment Generator Tests',
                'Tests for investment data generation and fraud detection'
            )
        }
        
    def run(self, test):
        """Run the tests with timing information."""
        self.start_time = time.time()
        result = super().run(test)
        elapsed = time.time() - self.start_time
        
        # Print section summaries
        self._print_section_summaries()
        
        self.stream.writeln(colored(f"\nTotal time elapsed: {elapsed:.2f}s", 'cyan'))
        return result
    
    def _print_section_summaries(self):
        """Print summaries for all test sections."""
        for section in self.sections.values():
            if section.total > 0:  # Only show sections that had tests run
                print(colored(f"\n╔══ {section.name} ══╗", 'cyan', attrs=['bold']))
                print(colored(f"Description: {section.description}", 'white'))
                print(colored("─" * 50, 'cyan'))
                print(colored(f"Total Tests: {section.total}", 'white'))
                print(colored(f"Passed: {section.passed} ✓", 'green'))
                if section.failed > 0:
                    print(colored(f"Failed: {section.failed} ✗", 'red'))
                if section.errors > 0:
                    print(colored(f"Errors: {section.errors} ⚠", 'red'))
                if section.skipped > 0:
                    print(colored(f"Skipped: {section.skipped} ⚡", 'yellow'))
                print(colored(f"Time: {section.elapsed_time:.2f}s", 'cyan'))
                
                # Print failures for this section if any
                if section.failures:
                    print(colored("\nFailures in this section:", 'red'))
                    for i, (test, trace) in enumerate(section.failures, 1):
                        print(colored(f"{i}. {test.id()}", 'red'))
                
                print(colored("╚" + "═" * 48 + "╝", 'cyan', attrs=['bold']))
        
class ColoredOutputWrapper:
    """Wrapper for stream output to add colors."""
    
    def __init__(self, stream):
        self.stream = stream
        self._current_test = None
        self._current_class = None
        self._failure_count = 0
        self._current_section = None
        
    def write(self, text: str) -> None:
        """Write colored output based on test results."""
        # Handle test case headers
        if " (" in text and "..." in text:
            test_name = text.split(" (")[0]
            class_name = text.split("(")[1].split(")")[0].split(".")[-1]
            self._current_test = test_name
            self._current_class = class_name
            self._current_section = self.get_runner().sections.get(class_name)
            
            if self._current_section:
                self._current_section.total += 1
                if not self._current_section.start_time:
                    self._current_section.start_time = time.time()
            
            # Get the test method's docstring
            test_class = globals()[class_name]
            test_method = getattr(test_class, test_name)
            docstring = inspect.getdoc(test_method) or "No description available"
            
            # Print test header with description
            self.stream.write("\n" + "─" * 80 + "\n")
            if self._current_section:
                self.stream.write(colored(f"Section: ", 'blue', attrs=['bold']) + 
                                colored(self._current_section.name, 'blue') + "\n")
            self.stream.write(colored(f"Test: ", 'blue', attrs=['bold']) + 
                            colored(test_name, 'blue') + "\n")
            self.stream.write(colored("Description: ", 'blue', attrs=['bold']) + 
                            colored(docstring, 'white') + "\n")
            self.stream.write("─" * 80 + "\n")
            return
            
        # Handle test results
        if text.strip() == "ok":
            if self._current_section:
                self._current_section.passed += 1
                self._current_section.end_time = time.time()
            self.stream.write(colored("✓ PASS", 'green'))
        elif text.strip() == "FAIL":
            self._failure_count += 1
            if self._current_section:
                self._current_section.failed += 1
                self._current_section.end_time = time.time()
            self.stream.write(colored("✗ FAIL", 'red', attrs=['bold']))
        elif text.strip() == "ERROR":
            self._failure_count += 1
            if self._current_section:
                self._current_section.errors += 1
                self._current_section.end_time = time.time()
            self.stream.write(colored("⚠ ERROR", 'red', attrs=['bold']))
        elif text.strip() == "skipped":
            if self._current_section:
                self._current_section.skipped += 1
                self._current_section.end_time = time.time()
            self.stream.write(colored("⚡ SKIP", 'yellow'))
        # Handle failure details
        elif text.startswith("======================================================================"):
            self.stream.write("\n" + colored(f"Failure #{self._failure_count} Details:", 'red', attrs=['bold']) + "\n")
            self.stream.write(colored("=" * 80 + "\n", 'red'))
        elif text.startswith("----------------------------------------------------------------------"):
            self.stream.write(colored("-" * 80 + "\n", 'yellow'))
        elif text.startswith("Traceback"):
            self.stream.write(colored("\nStacktrace:", 'red', attrs=['bold']) + "\n")
            self.stream.write(colored(text, 'red'))
        elif "Error" in text or "Exception" in text:
            self.stream.write(colored("\nError Message:", 'red', attrs=['bold']) + "\n")
            self.stream.write(colored(text, 'red'))
        elif text.strip().startswith("Ran"):
            self.stream.write("\n" + "═" * 80 + "\n")
            self.stream.write(colored(text, 'cyan'))
        else:
            # For assertion errors, try to provide more context
            if "AssertionError" in text:
                self.stream.write(colored("\nAssertion Details:", 'yellow', attrs=['bold']) + "\n")
                # Try to extract expected vs actual values
                if "!=" in text:
                    expected, actual = text.split("!=")
                    self.stream.write(colored("Expected: ", 'yellow') + expected.strip() + "\n")
                    self.stream.write(colored("Actual:   ", 'yellow') + actual.strip() + "\n")
                else:
                    self.stream.write(colored(text, 'yellow'))
            else:
                self.stream.write(text)
    
    def get_runner(self):
        """Get the test runner instance."""
        return unittest.TestResult.test_runner
            
    def writeln(self, text: Optional[str] = None) -> None:
        """Write a line of text."""
        if text is not None:
            self.write(text)
        self.write('\n')
        
    def flush(self) -> None:
        """Flush the stream."""
        self.stream.flush()

def create_test_suite() -> unittest.TestSuite:
    """Create a test suite containing all tests.
    
    Returns:
        unittest.TestSuite: Test suite with all test cases
    """
    test_suite = unittest.TestSuite()
    
    # Add test cases from each test file
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatasetGeneratorClass))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmploymentGenerator))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInvestmentGenerator))
    
    return test_suite

def run_tests() -> None:
    """Run all tests with colored output."""
    # Print header
    print(colored("\n╔══ Synthetic Data Generator Test Suite ══╗", 'cyan', attrs=['bold']))
    print(colored("║", 'cyan', attrs=['bold']) + " " * 36 + colored("║", 'cyan', attrs=['bold']))
    print(colored("╚══════════════════════════════════════╝\n", 'cyan', attrs=['bold']))
    
    # Create and run test suite
    suite = create_test_suite()
    runner = ColoredTestRunner(verbosity=2)
    result = runner.run(suite)

if __name__ == '__main__':
    run_tests() 