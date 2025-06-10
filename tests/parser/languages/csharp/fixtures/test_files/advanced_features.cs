using System;

namespace MyApplication.Events
{
    /// <summary>
    /// Event arguments for notification events
    /// </summary>
    public class NotificationEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the notification message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Gets the notification timestamp
        /// </summary>
        public DateTime Timestamp { get; }

        /// <summary>
        /// Initializes notification event arguments
        /// </summary>
        /// <param name="message">The notification message</param>
        public NotificationEventArgs(string message)
        {
            Message = message;
            Timestamp = DateTime.Now;
        }
    }

    /// <summary>
    /// Delegate for handling custom operations
    /// </summary>
    /// <param name="input">Input parameter</param>
    /// <returns>Operation result</returns>
    public delegate string CustomOperation(string input);

    /// <summary>
    /// Service class demonstrating events and delegates
    /// </summary>
    public class NotificationService
    {
        private string _serviceName = "DefaultService";

        /// <summary>
        /// Event fired when a notification is sent
        /// </summary>
        public event EventHandler<NotificationEventArgs>? NotificationSent;

        /// <summary>
        /// Gets or sets the service name with validation
        /// </summary>
        public string ServiceName 
        { 
            get => _serviceName;
            set 
            {
                if (string.IsNullOrEmpty(value))
                    throw new ArgumentException("Service name cannot be null or empty");
                _serviceName = value;
            }
        }

        /// <summary>
        /// Auto-implemented property for service status
        /// </summary>
        public bool IsActive { get; set; } = true;

        /// <summary>
        /// Sends a notification message
        /// </summary>
        /// <param name="message">The message to send</param>
        public void SendNotification(string message)
        {
            if (!IsActive)
                return;

            var args = new NotificationEventArgs(message);
            OnNotificationSent(args);
        }

        /// <summary>
        /// Processes input using a custom operation delegate
        /// </summary>
        /// <param name="input">Input to process</param>
        /// <param name="operation">Custom operation to apply</param>
        /// <returns>Processed result</returns>
        public string ProcessWithCustomOperation(string input, CustomOperation operation)
        {
            return operation?.Invoke(input) ?? input;
        }

        /// <summary>
        /// Raises the NotificationSent event
        /// </summary>
        /// <param name="e">Event arguments</param>
        protected virtual void OnNotificationSent(NotificationEventArgs e)
        {
            NotificationSent?.Invoke(this, e);
        }
    }
}