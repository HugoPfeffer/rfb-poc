"""Utility classes and functions for testing."""
from typing import List, Tuple
import unittest
import pytest

class TestSection:
    """Represents a section of tests. This is a helper class, not a test class."""
    
    __test__ = False  # Tell pytest to ignore this class
    
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