using System;

// Top-level statements (C# 9+)
Console.WriteLine("Hello from top-level program!");

/// <summary>
/// A record type representing a person
/// </summary>
/// <param name="FirstName">The person's first name</param>
/// <param name="LastName">The person's last name</param>
/// <param name="Age">The person's age</param>
public record Person(string FirstName, string LastName, int Age)
{
    /// <summary>
    /// Gets the full name of the person
    /// </summary>
    public string FullName => $"{FirstName} {LastName}";

    /// <summary>
    /// Checks if the person is an adult
    /// </summary>
    /// <returns>True if age is 18 or above</returns>
    public bool IsAdult() => Age >= 18;
}

/// <summary>
/// A record struct for coordinates
/// </summary>
/// <param name="X">X coordinate</param>
/// <param name="Y">Y coordinate</param>
public readonly record struct Point(double X, double Y)
{
    /// <summary>
    /// Calculates distance from origin
    /// </summary>
    /// <returns>Distance from (0,0)</returns>
    public double DistanceFromOrigin() => Math.Sqrt(X * X + Y * Y);
}

/// <summary>
/// Class demonstrating pattern matching and switch expressions
/// </summary>
public static class PatternMatchingExample
{
    /// <summary>
    /// Describes an object using pattern matching
    /// </summary>
    /// <param name="obj">Object to describe</param>
    /// <returns>Description string</returns>
    public static string DescribeObject(object obj) => obj switch
    {
        string s when s.Length > 10 => "Long string",
        string s => $"String: {s}",
        int i when i > 0 => "Positive integer",
        int i => "Non-positive integer",
        Person p => $"Person: {p.FullName}",
        null => "Null object",
        _ => "Unknown object type"
    };
}