using System;
using System.Threading.Tasks;

namespace MyApplication.Advanced
{
    /// <summary>
    /// Outer class demonstrating nesting capabilities
    /// </summary>
    public class OuterClass
    {
        /// <summary>
        /// Public field in outer class
        /// </summary>
        public string OuterField = "Outer";

        /// <summary>
        /// Method in outer class
        /// </summary>
        /// <returns>A message from outer class</returns>
        public string OuterMethod()
        {
            return "Message from outer class";
        }

        /// <summary>
        /// Nested class within OuterClass
        /// </summary>
        public class NestedClass
        {
            /// <summary>
            /// Method in nested class
            /// </summary>
            /// <returns>A message from nested class</returns>
            public string NestedMethod()
            {
                return "Message from nested class";
            }
        }

        /// <summary>
        /// Private nested class
        /// </summary>
        private class PrivateNested
        {
            /// <summary>
            /// Private method in private nested class
            /// </summary>
            /// <returns>Private message</returns>
            private string PrivateMethod()
            {
                return "Private message";
            }
        }
    }

    /// <summary>
    /// Static class with extension methods
    /// </summary>
    public static class StringExtensions
    {
        /// <summary>
        /// Checks if a string is null or empty
        /// </summary>
        /// <param name="value">The string to check</param>
        /// <returns>True if null or empty</returns>
        public static bool IsNullOrEmpty(this string value)
        {
            return string.IsNullOrEmpty(value);
        }

        /// <summary>
        /// Reverses a string
        /// </summary>
        /// <param name="value">The string to reverse</param>
        /// <returns>The reversed string</returns>
        public static string Reverse(this string value)
        {
            if (value.IsNullOrEmpty())
                return value;
            
            char[] charArray = value.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }
    }
}