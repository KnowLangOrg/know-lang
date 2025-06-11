"""Expected test results for C# parser tests"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ChunkExpectation:
    """Expected values for a code chunk"""
    name: str
    docstring: str
    content_snippet: str
    namespace: Optional[str] = None
    class_name: Optional[str] = None


def get_test_file_path(filename: str) -> Path:
    """Get the path to a test file"""
    test_files_dir = Path(__file__).parent / 'test_files'
    return test_files_dir / filename


def get_test_file_content(filename: str) -> str:
    """Get the content of a test file"""
    file_path = get_test_file_path(filename)
    return file_path.read_text(encoding='utf-8')


def get_all_test_files() -> list[Path]:
    """Get all test files in the test_files directory"""
    test_files_dir = Path(__file__).parent / 'test_files'
    return list(test_files_dir.glob('*.cs'))


# Test expectations for each file
SIMPLE_FILE_EXPECTATIONS = {
    'HelloWorld': ChunkExpectation(
        name='HelloWorld',
        docstring='A simple hello world class',
        content_snippet='public class HelloWorld'
    ),
    'Person': ChunkExpectation(
        name='Person',
        docstring='A simple data class',
        content_snippet='public class Person'
    ),
}

COMPLEX_FILE_EXPECTATIONS = {
    'DeprecatedAttribute': ChunkExpectation(
        name='DeprecatedAttribute',
        docstring='Marks a class as deprecated with a custom message',
        content_snippet='public class DeprecatedAttribute : Attribute',
        namespace='MyApplication.Services'
    ),
    'IRepository': ChunkExpectation(
        name='IRepository',
        docstring='Generic repository interface for data access',
        content_snippet='public interface IRepository<T> where T : class',
        namespace='MyApplication.Services'
    ),
    'GenericRepository': ChunkExpectation(
        name='GenericRepository',
        docstring='Generic repository implementation',
        content_snippet='public class GenericRepository<T> : IRepository<T>',
        namespace='MyApplication.Services'
    ),
    'GetAllAsync': ChunkExpectation(
        name='GetAllAsync',
        docstring='Gets all entities from the in-memory collection',
        content_snippet='public async Task<IEnumerable<T>> GetAllAsync()',
        namespace='MyApplication.Services',
        class_name='GenericRepository'
    ),
    'GetByIdAsync': ChunkExpectation(
        name='GetByIdAsync',
        docstring='Gets an entity by ID (simplified implementation)',
        content_snippet='public async Task<T?> GetByIdAsync(int id)',
        namespace='MyApplication.Services',
        class_name='GenericRepository'
    ),
    'AddAsync': ChunkExpectation(
        name='AddAsync',
        docstring='Adds an entity to the collection',
        content_snippet='public async Task<T> AddAsync(T entity)',
        namespace='MyApplication.Services',
        class_name='GenericRepository'
    ),
    'User': ChunkExpectation(
        name='User',
        docstring='Represents a user in the system',
        content_snippet='public class User',
        namespace='MyApplication.Models'
    ),
    'IsValid': ChunkExpectation(
        name='IsValid',
        docstring='Validates the user data',
        content_snippet='public bool IsValid()',
        namespace='MyApplication.Models',
        class_name='User'
    )
}

NESTED_FILE_EXPECTATIONS = {
    'OuterClass': ChunkExpectation(
        name='OuterClass',
        docstring='Outer class demonstrating nesting capabilities',
        content_snippet='public class OuterClass',
        namespace='MyApplication.Advanced'
    ),
    'OuterMethod': ChunkExpectation(
        name='OuterMethod',
        docstring='Method in outer class',
        content_snippet='public string OuterMethod()',
        namespace='MyApplication.Advanced',
        class_name='OuterClass'
    ),
    'StringExtensions': ChunkExpectation(
        name='StringExtensions',
        docstring='Static class with extension methods',
        content_snippet='public static class StringExtensions',
        namespace='MyApplication.Advanced'
    ),
    'IsNullOrEmpty': ChunkExpectation(
        name='IsNullOrEmpty',
        docstring='Checks if a string is null or empty',
        content_snippet='public static bool IsNullOrEmpty(this string value)',
        namespace='MyApplication.Advanced',
        class_name='StringExtensions'
    ),
    'Reverse': ChunkExpectation(
        name='Reverse',
        docstring='Reverses a string',
        content_snippet='public static string Reverse(this string value)',
        namespace='MyApplication.Advanced',
        class_name='StringExtensions'
    )
}

ADVANCED_FEATURES_EXPECTATIONS = {
    'NotificationEventArgs': ChunkExpectation(
        name='NotificationEventArgs',
        docstring='Event arguments for notification events',
        content_snippet='public class NotificationEventArgs : EventArgs',
        namespace='MyApplication.Events'
    ),
    'NotificationService': ChunkExpectation(
        name='NotificationService',
        docstring='Service class demonstrating events and delegates',
        content_snippet='public class NotificationService',
        namespace='MyApplication.Events'
    ),
    'SendNotification': ChunkExpectation(
        name='SendNotification',
        docstring='Sends a notification message',
        content_snippet='public void SendNotification(string message)',
        namespace='MyApplication.Events',
        class_name='NotificationService'
    ),
    'ProcessWithCustomOperation': ChunkExpectation(
        name='ProcessWithCustomOperation',
        docstring='Processes input using a custom operation delegate',
        content_snippet='public string ProcessWithCustomOperation(string input, CustomOperation operation)',
        namespace='MyApplication.Events',
        class_name='NotificationService'
    )
}

MODERN_CSHARP_EXPECTATIONS = {
    'Person': ChunkExpectation(
        name='Person',
        docstring='A record type representing a person',
        content_snippet='public record Person(string FirstName, string LastName, int Age)'
    ),
    'IsAdult': ChunkExpectation(
        name='IsAdult',
        docstring='Checks if the person is an adult',
        content_snippet='public bool IsAdult() => Age >= 18;',
        class_name='Person'
    ),
    'Point': ChunkExpectation(
        name='Point',
        docstring='A record struct for coordinates',
        content_snippet='public readonly record struct Point(double X, double Y)'
    ),
    'DistanceFromOrigin': ChunkExpectation(
        name='DistanceFromOrigin',
        docstring='Calculates distance from origin',
        content_snippet='public double DistanceFromOrigin() => Math.Sqrt(X * X + Y * Y);',
        class_name='Point'
    ),
    'PatternMatchingExample': ChunkExpectation(
        name='PatternMatchingExample',
        docstring='Class demonstrating pattern matching and switch expressions',
        content_snippet='public static class PatternMatchingExample'
    ),
    'DescribeObject': ChunkExpectation(
        name='DescribeObject',
        docstring='Describes an object using pattern matching',
        content_snippet='public static string DescribeObject(object obj) => obj switch',
        class_name='PatternMatchingExample'
    )
}

# All expectations combined for easy access
ALL_EXPECTATIONS: Dict[str, Dict[str, ChunkExpectation]] = {
    'simple.cs': SIMPLE_FILE_EXPECTATIONS,
    'complex.cs': COMPLEX_FILE_EXPECTATIONS,
    'nested.cs': NESTED_FILE_EXPECTATIONS,
    'advanced_features.cs': ADVANCED_FEATURES_EXPECTATIONS,
    'modern.cs': MODERN_CSHARP_EXPECTATIONS
}


def load_test_expectations() -> Dict[str, Dict[str, ChunkExpectation]]:
    """Load all test expectations"""
    return ALL_EXPECTATIONS


def get_expectations_for_file(filename: str) -> Dict[str, ChunkExpectation]:
    """Get expectations for a specific test file"""
    return ALL_EXPECTATIONS.get(filename, {})
