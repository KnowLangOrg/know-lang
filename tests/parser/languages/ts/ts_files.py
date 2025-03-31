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
 * Marks a class as deprecated
 */
function deprecated(message: string) {
    return (target: any) => {
        console.warn(`WARNING: ${target.name} is deprecated. ${message}`);
    };
}

/**
 * A decorated class at the top level
 */
@deprecated("Use NewService instead")
class DecoratedService {
    getData(): string {
        return "Service data";
    }
}

namespace Utils {
    /**
     * Generic repository for data access
     * @typeparam T The entity type
     */
    class Repository<T> {
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

# Simple React component with JSX (TSX)
SIMPLE_TSX = """
import React, { useState } from 'react';

/**
 * Props for Button component
 */
interface ButtonProps {
  onClick: () => void;
  label: string;
  disabled?: boolean;
}

/**
 * A simple button component
 */
const Button: React.FC<ButtonProps> = ({ onClick, label, disabled = false }) => {
  return (
    <button 
      onClick={onClick} 
      disabled={disabled}
      className="primary-button"
    >
      {label}
    </button>
  );
};

export default Button;
"""

# Complex React component with JSX (TSX)
COMPLEX_TSX = """
import React, { useState, useEffect, useCallback } from 'react';

/**
 * User data structure
 */
interface User {
  id: number;
  name: string;
  email: string;
}

/**
 * Props for the UserList component
 */
interface UserListProps {
  initialUsers?: User[];
  onUserSelect?: (user: User) => void;
}

/**
 * A component that displays a list of users with filtering
 */
const UserList: React.FC<UserListProps> = ({ 
  initialUsers = [], 
  onUserSelect 
}) => {
  const [users, setUsers] = useState<User[]>(initialUsers);
  const [filter, setFilter] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  
  /**
   * Load users from API
   */
  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/users');
      const data = await response.json();
      setUsers(data);
    } catch (error) {
      console.error('Failed to load users:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Filter users by name
   */
  const filteredUsers = users.filter(user => 
    user.name.toLowerCase().includes(filter.toLowerCase())
  );
  
  /**
   * Handle user selection
   */
  const handleUserClick = (user: User) => {
    if (onUserSelect) {
      onUserSelect(user);
    }
  };
  
  useEffect(() => {
    if (initialUsers.length === 0) {
      loadUsers();
    }
  }, [initialUsers.length, loadUsers]);
  
  return (
    <div className="user-list-container">
      <div className="filter-container">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter users..."
          className="filter-input"
        />
      </div>
      
      {loading ? (
        <div className="loading">Loading users...</div>
      ) : (
        <ul className="user-list">
          {filteredUsers.map(user => (
            <li 
              key={user.id} 
              onClick={() => handleUserClick(user)}
              className="user-item"
            >
              <div className="user-name">{user.name}</div>
              <div className="user-email">{user.email}</div>
            </li>
          ))}
        </ul>
      )}
      
      {filteredUsers.length === 0 && !loading && (
        <div className="no-results">No users found</div>
      )}
    </div>
  );
};

export default UserList;
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

# Invalid TSX syntax
INVALID_TSX = """
import React from 'react';

const InvalidComponent = () => {
  return (
    <div>
      <h1>Hello,</h1>
      <p>This component has invalid JSX syntax
    </div>  // Missing closing tag for <p>
  );
};

export default InvalidComponent;
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
    'deprecated': ChunkExpectation(
        name='deprecated',
        docstring='Marks a class as deprecated',
        content_snippet='function deprecated(message: string)'
    ),
    'DecoratedService': ChunkExpectation(
        name='DecoratedService',
        docstring='A decorated class at the top level',
        content_snippet='@deprecated("Use NewService instead")\nclass DecoratedService {'
    ),
    'Repository': ChunkExpectation(
        name='Repository',
        docstring='Generic repository for data access\n@typeparam T The entity type',
        content_snippet='class Repository<T> {'
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

SIMPLE_TSX_EXPECTATIONS = {
    'ButtonProps': ChunkExpectation(
        name='ButtonProps',
        docstring='Props for Button component',
        content_snippet='interface ButtonProps {'
    ),
    'Button': ChunkExpectation(
        name='Button',
        docstring='A simple button component',
        content_snippet='const Button: React.FC<ButtonProps> = ({ onClick, label, disabled = false }) => {'
    )
}

COMPLEX_TSX_EXPECTATIONS = {
    'User': ChunkExpectation(
        name='User',
        docstring='User data structure',
        content_snippet='interface User {'
    ),
    'UserListProps': ChunkExpectation(
        name='UserListProps',
        docstring='Props for the UserList component',
        content_snippet='interface UserListProps {'
    ),
    'UserList': ChunkExpectation(
        name='UserList',
        docstring='A component that displays a list of users with filtering',
        content_snippet='const UserList: React.FC<UserListProps> = ({'
    ),
    'loadUsers': ChunkExpectation(
        name='loadUsers',
        docstring='Load users from API',
        content_snippet='const loadUsers = useCallback(async () => {'
    )
}

# Compile test files
TEST_FILES = {
    'simple.ts': SIMPLE_TS,
    'complex.ts': COMPLEX_TS,
    'simple.tsx': SIMPLE_TSX,
    'complex.tsx': COMPLEX_TSX,
    'invalid.ts': INVALID_TS,
    'invalid.tsx': INVALID_TSX
}