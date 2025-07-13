from typing import NamedTuple


class ExpectedChunk(NamedTuple):
    name: str
    docstring: str
    content_snippet: str  # A unique part of the content that should be present


# Test file contents
SIMPLE_FUCNTION_CLEANED_DOCSTRING = "Say hello to the world"
SIMPLE_FUNCTION_DOCSTRING = f'"""{SIMPLE_FUCNTION_CLEANED_DOCSTRING}"""'
SIMPLE_FUNCTION = f"""
def hello_world():
    {SIMPLE_FUNCTION_DOCSTRING}
    return "Hello, World!"
"""

SIMPLE_CLASS_DOCSTRING = '"""A simple class for testing"""'
SIMPLE_CLASS = f"""
class SimpleClass:
    {SIMPLE_CLASS_DOCSTRING}
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""

NESTED_OUTER_CLASS_DOCSTRING = "#Outer class docstring"
NESTED_CLASS = f'''
class OuterClass:
    {NESTED_OUTER_CLASS_DOCSTRING}
    class InnerClass:
        """Inner class docstring"""
        def inner_method(self):
            return "inner"

    def outer_method(self):
        return "outer"
'''


COMPLEX_FUNCTION_DOCSTRING = """\"\"\"
A complex function with type hints and docstring
Args:
    param1: First parameter
    param2: Optional second parameter
Returns:
    List of strings
\"\"\""""

COMPLEX_CLASS_DOCSTRING = '"""Complex class implementation"""'
COMPLEX_FILE = f'''
import os
from typing import List, Optional

def complex_function(param1: str, param2: Optional[int] = None) -> List[str]:
    {COMPLEX_FUNCTION_DOCSTRING}
    results = []
    if param2 is not None:
        results.extend([param1] * param2)
    return results

# Some comment
CONSTANT = 42

class ComplexClass:
    {COMPLEX_CLASS_DOCSTRING}
    """Complex class implementation Test Test"""
    def __init__(self):
        self._value = None
'''

INVALID_SYNTAX = """def invalid_syntax(:"""

# Expected test results
SIMPLE_FILE_EXPECTATIONS = {
    "hello_world": ExpectedChunk(
        name="hello_world",
        docstring=SIMPLE_FUNCTION_DOCSTRING,
        content_snippet='return "Hello, World!"',
    ),
    "SimpleClass": ExpectedChunk(
        name="SimpleClass",
        docstring=SIMPLE_CLASS_DOCSTRING,
        content_snippet="self.value = 42",
    ),
}

NESTED_CLASS_EXPECTATIONS = {
    "OuterClass": ExpectedChunk(
        name="OuterClass",
        docstring=NESTED_OUTER_CLASS_DOCSTRING,
        content_snippet="class InnerClass",
    )
}

COMPLEX_FILE_EXPECTATIONS = {
    "complex_function": ExpectedChunk(
        name="complex_function",
        docstring=COMPLEX_FUNCTION_DOCSTRING,
        content_snippet="List[str]",
    ),
    "ComplexClass": ExpectedChunk(
        name="ComplexClass",
        docstring=COMPLEX_CLASS_DOCSTRING,
        content_snippet="_value = None",
    ),
}

DECORATED_FUNCTION_DOCSTRING = '"""A function with decorators"""'
DECORATED_CLASS_DOCSTRING = '"""A class with decorators"""'
DECORATED_FILE = f"""
@my_decorator
@another_decorator
def decorated_function():
    {DECORATED_FUNCTION_DOCSTRING}
    return "I am decorated"

@my_decorator
class DecoratedClass:
    {DECORATED_CLASS_DOCSTRING}
    def __init__(self):
        self.value = "decorated"
"""

# Add expectations for decorated components
DECORATED_FILE_EXPECTATIONS = {
    "decorated_function": ExpectedChunk(
        name="decorated_function",
        docstring=DECORATED_FUNCTION_DOCSTRING,
        content_snippet='return "I am decorated"',
    ),
    "DecoratedClass": ExpectedChunk(
        name="DecoratedClass",
        docstring=DECORATED_CLASS_DOCSTRING,
        content_snippet='self.value = "decorated"',
    ),
}

TEST_FILES = {
    "simple.py": SIMPLE_FUNCTION + SIMPLE_CLASS,
    "nested.py": NESTED_CLASS,
    "complex.py": COMPLEX_FILE,
    "decorated.py": DECORATED_FILE,
}
