/// <summary>
/// A simple hello world class
/// </summary>
public class HelloWorld
{
    /// <summary>
    /// Says hello to the world
    /// </summary>
    /// <returns>A greeting message</returns>
    public string SayHello()
    {
        return "Hello, World!";
    }

    /// <summary>
    /// Says hello to a specific person
    /// </summary>
    /// <param name="name">The name of the person to greet</param>
    /// <returns>A personalized greeting</returns>
    public string SayHello(string name)
    {
        return $"Hello, {name}!";
    }

    // Method without XML documentation
    public void DoSomething()
    {
        Console.WriteLine("Doing something...");
    }
}

/// <summary>
/// A simple data class
/// </summary>
public class Person
{
    /// <summary>
    /// Gets or sets the person's name
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Gets or sets the person's age
    /// </summary>
    public int Age { get; set; }

    /// <summary>
    /// Gets the person's full information
    /// </summary>
    /// <returns>Formatted person information</returns>
    public string GetInfo()
    {
        return $"{Name} is {Age} years old";
    }
}