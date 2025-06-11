// This file contains intentional syntax errors for testing

namespace InvalidNamespace
{
    /// <summary>Missing closing brace for class</summary>
    public class IncompleteClass
    {
        public void ValidMethod()
        {
            return "This is fine";
        }
    // Missing closing brace for class

    /// <summary>Method with syntax error</summary>
    public class AnotherClass
    {
        // Invalid method signature - missing parameter type
        public void InvalidMethod(string name, invalidParam)
        {
            Console.WriteLine("This won't compile");
        }

        // Missing return type
        public InvalidReturnType()
        {
            return "Should be string";
        }
    }
}