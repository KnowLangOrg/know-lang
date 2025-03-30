from dataclasses import dataclass


@dataclass
class ChunkExpectation:
    """Expected values for a code chunk"""
    name: str
    docstring: str
    content_snippet: str


# Simple TypeScript file with functions, classes, interfaces, and type aliases
SIMPLE_TS = """
/**
 * A simple hello world function
 * @param name Name to greet
 * @returns Greeting message
 */
function helloWorld(name: string): string {
    return `Hello, ${name}!`;
}

/**
 * A simple counter class
 */
class Counter {
    private count: number = 0;
    
    /**
     * Increment the counter
     * @returns The new count
     */
    increment(): number {
        return ++this.count;
    }
    
    /**
     * Get the current count
     */
    get value(): number {
        return this.count;
    }
}

/**
 * Person interface
 */
interface Person {
    name: string;
    age: number;
    greet(): string;
}

/**
 * A type alias for a user object
 */
type User = {
    id: number;
    username: string;
    isAdmin: boolean;
};
"""

# Complex TypeScript file with generics, namespaces, and decorators
COMPLEX_TS = """
/**
 * Logging decorator
 */
function log(target: any, key: string, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    descriptor.value = function(...args: any[]) {
        console.log(`Calling ${key} with args: ${JSON.stringify(args)}`);
        return original.apply(this, args);
    };
    return descriptor;
}

namespace Utils {
    /**
     * Generic repository for data access
     * @typeparam T The entity type
     */
    export class Repository<T> {
        items: T[] = [];
        
        /**
         * Add an item to the repository
         * @param item Item to add
         */
        add(item: T): void {
            this.items.push(item);
        }
        
        /**
         * Get all items
         * @returns All items in the repository
         */
        @log
        getAll(): T[] {
            return [...this.items];
        }
    }
}

/**
 * Configuration options for an API client
 */
interface ApiConfig {
    baseUrl: string;
    timeout?: number;
    headers?: Record<string, string>;
}

/**
 * Generic API client
 * @typeparam T Response data type
 * @typeparam U Request data type
 */
class ApiClient<T, U = any> {
    private config: ApiConfig;
    
    constructor(config: ApiConfig) {
        this.config = config;
    }
    
    /**
     * Send a request
     * @param data Request data
     * @returns Promise with response
     */
    async request(data: U): Promise<T> {
        // Implementation details
        return {} as T;
    }
}

/**
 * Result type for API responses
 */
type ApiResult<T> = {
    success: boolean;
    data?: T;
    error?: string;
};

/**
 * Function with arrow syntax
 */
const fetchData = async <T>(url: string): Promise<T> => {
    // Implementation details
    return {} as T;
};
"""

# TypeScript React component
REACT_TS = """
import React, { useState, useEffect } from 'react';

/**
 * Props for the Counter component
 */
interface CounterProps {
    initialValue?: number;
    step?: number;
}

/**
 * A simple counter component
 */
const Counter: React.FC<CounterProps> = ({ initialValue = 0, step = 1 }) => {
    const [count, setCount] = useState(initialValue);
    
    /**
     * Increment the counter
     */
    const increment = () => {
        setCount(prev => prev + step);
    };
    
    /**
     * Decrement the counter
     */
    const decrement = () => {
        setCount(prev => prev - step);
    };
    
    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);
    
    return (
        <div>
            <h1>Count: {count}</h1>
            <button onClick={increment}>+</button>
            <button onClick={decrement}>-</button>
        </div>
    );
};

export default Counter;
"""

# Invalid TypeScript syntax
INVALID_TS = """
class InvalidClass {
    constructor(public name string) {} // Missing colon
    
    broken method() { // Missing parentheses
        return 'broken';
    }
}
"""

# Test file expectations
SIMPLE_FILE_EXPECTATIONS = {
    'helloWorld': ChunkExpectation(
        name='helloWorld',
        docstring='A simple hello world function\n@param name Name to greet\n@returns Greeting message',
        content_snippet='function helloWorld(name: string): string {'
    ),
    'Counter': ChunkExpectation(
        name='Counter',
        docstring='A simple counter class',
        content_snippet='class Counter {'
    ),
    'increment': ChunkExpectation(
        name='increment',
        docstring='Increment the counter\n@returns The new count',
        content_snippet='increment(): number {'
    ),
    'Person': ChunkExpectation(
        name='Person',
        docstring='Person interface',
        content_snippet='interface Person {'
    ),
    'User': ChunkExpectation(
        name='User',
        docstring='A type alias for a user object',
        content_snippet='type User = {'
    )
}

COMPLEX_FILE_EXPECTATIONS = {
    'log': ChunkExpectation(
        name='log',
        docstring='Logging decorator',
        content_snippet='function log(target: any, key: string, descriptor: PropertyDescriptor)'
    ),
    'Repository': ChunkExpectation(
        name='Repository',
        docstring='Generic repository for data access\n@typeparam T The entity type',
        content_snippet='export class Repository<T> {'
    ),
    'getAll': ChunkExpectation(
        name='getAll',
        docstring='Get all items\n@returns All items in the repository',
        content_snippet='@log\n        getAll(): T[] {'
    ),
    'ApiClient': ChunkExpectation(
        name='ApiClient',
        docstring='Generic API client\n@typeparam T Response data type\n@typeparam U Request data type',
        content_snippet='class ApiClient<T, U = any> {'
    ),
    'ApiConfig': ChunkExpectation(
        name='ApiConfig',
        docstring='Configuration options for an API client',
        content_snippet='interface ApiConfig {'
    ),
    'ApiResult': ChunkExpectation(
        name='ApiResult',
        docstring='Result type for API responses',
        content_snippet='type ApiResult<T> = {'
    ),
    'fetchData': ChunkExpectation(
        name='fetchData',
        docstring='Function with arrow syntax',
        content_snippet='const fetchData = async <T>(url: string): Promise<T> => {'
    )
}

REACT_FILE_EXPECTATIONS = {
    'CounterProps': ChunkExpectation(
        name='CounterProps',
        docstring='Props for the Counter component',
        content_snippet='interface CounterProps {'
    ),
    'Counter': ChunkExpectation(
        name='Counter',
        docstring='A simple counter component',
        content_snippet='const Counter: React.FC<CounterProps> = ({ initialValue = 0, step = 1 }) => {'
    )
}

# Compile test files
TEST_FILES = {
    'simple.ts': SIMPLE_TS,
    'complex.ts': COMPLEX_TS,
    'component.tsx': REACT_TS,
    'invalid.ts': INVALID_TS
}