using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace MyApplication.Services
{
    /// <summary>
    /// Marks a class as deprecated with a custom message
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
    public class DeprecatedAttribute : Attribute
    {
        /// <summary>
        /// Gets the deprecation message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Initializes a new instance of the DeprecatedAttribute
        /// </summary>
        /// <param name="message">The deprecation message</param>
        public DeprecatedAttribute(string message)
        {
            Message = message;
        }
    }

    /// <summary>
    /// Generic repository interface for data access
    /// </summary>
    /// <typeparam name="T">The entity type</typeparam>
    public interface IRepository<T> where T : class
    {
        /// <summary>
        /// Gets all entities
        /// </summary>
        /// <returns>Collection of all entities</returns>
        Task<IEnumerable<T>> GetAllAsync();

        /// <summary>
        /// Gets an entity by its identifier
        /// </summary>
        /// <param name="id">The entity identifier</param>
        /// <returns>The entity if found, null otherwise</returns>
        Task<T?> GetByIdAsync(int id);

        /// <summary>
        /// Adds a new entity
        /// </summary>
        /// <param name="entity">The entity to add</param>
        /// <returns>The added entity</returns>
        Task<T> AddAsync(T entity);
    }

    /// <summary>
    /// Generic repository implementation
    /// </summary>
    /// <typeparam name="T">The entity type</typeparam>
    [Deprecated("Use EntityFramework repository instead")]
    public class GenericRepository<T> : IRepository<T> where T : class
    {
        private readonly List<T> _items = new();

        /// <summary>
        /// Gets all entities from the in-memory collection
        /// </summary>
        /// <returns>All entities</returns>
        public async Task<IEnumerable<T>> GetAllAsync()
        {
            await Task.Delay(1); // Simulate async operation
            return _items.ToList();
        }

        /// <summary>
        /// Gets an entity by ID (simplified implementation)
        /// </summary>
        /// <param name="id">The entity ID</param>
        /// <returns>The entity if found</returns>
        public async Task<T?> GetByIdAsync(int id)
        {
            await Task.Delay(1);
            return _items.ElementAtOrDefault(id);
        }

        /// <summary>
        /// Adds an entity to the collection
        /// </summary>
        /// <param name="entity">Entity to add</param>
        /// <returns>The added entity</returns>
        public async Task<T> AddAsync(T entity)
        {
            await Task.Delay(1);
            _items.Add(entity);
            return entity;
        }
    }
}

namespace MyApplication.Models
{
    /// <summary>
    /// Represents a user in the system
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user ID
        /// </summary>
        [Required]
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the username
        /// </summary>
        [Required]
        [StringLength(50)]
        public string Username { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the user's email address
        /// </summary>
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        /// <summary>
        /// Validates the user data
        /// </summary>
        /// <returns>True if valid, false otherwise</returns>
        public bool IsValid()
        {
            return !string.IsNullOrEmpty(Username) && 
                   !string.IsNullOrEmpty(Email) &&
                   Email.Contains("@");
        }
    }
}