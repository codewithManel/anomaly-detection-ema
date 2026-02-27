# This Python 3.9.0 script was created by Adjaout Manel implements a real-time anomaly detection system using 
# an Exponential Moving Average (EMA) algorithm. The system processes a 
# simulated data stream with seasonal variations and random noise, and 
# identifies anomalies based on dynamic thresholding. The stream and anomalies
# are visualized in real-time using Matplotlib. 

# The code is optimized for both speed and efficiency, with robust error 
# handling and data validation in place to ensure reliability.
# Credits by: ADJAOUT Manel

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate new data with seasonal variation and noise
def generate_new_data_point(size=1, noise_level=0.5):
    """Generate a data point with seasonal variation and random noise."""
    
    # Create a seasonal component using a sine wave for seasonality
    seasonal_component = np.sin(np.linspace(0, 10, size))
    
    # Generate random noise and add it to the seasonal component
    data_point = np.random.normal(0, noise_level, size) + seasonal_component
    
    return data_point

# 2. Anomaly Detector based on Exponential Moving Average (EMA)
class EMAAnomalyDetector:
    """
    This class implements an anomaly detection algorithm using Exponential Moving Average (EMA).
    
    Attributes:
        alpha (float): Smoothing factor for EMA (0 < alpha <= 1).
        threshold (float): Threshold for detecting anomalies.
        history_size (int): Number of recent deviations to consider for dynamic thresholding.
        ema (float or None): Current EMA value.
        recent_deviations (list): List of recent deviations from the EMA.
    """
    
#NOTE:
# The threshold value plays a significant role in determining what constitutes an anomaly.
# Lower thresholds, such as 2.0, increase sensitivity, capturing minor variations in the data, 
 # which may be important but could also introduce more false positives.
 # A higher threshold, like 3.0, focuses on significant deviations, reducing false positives 
 # and only flagging larger outliers, which can be more useful for decision-making.
 #
# In this implementation, we use a threshold of 2.7 to balance both aspects.
# This value stays between 2 and 3, capturing relevant minor variations while also 
# reducing false positives, thus focusing on genuine outliers that are more relevant for 
# our specific decision-making needs.
    
#NOTE:
# The history size plays an important role in anomaly detection, as it defines the number of recent data points 
# considered when calculating the dynamic threshold. A larger history size, such as 150, provides more context 
# to the detector, allowing it to better understand trends and deviations in the data stream. This helps in detecting 
# anomalies that might be subtle or missed with smaller history sizes. 
# 
# After testing, a history size of 150 was chosen as it offers a good balance between sensitivity and false positives, 
# and provides sufficient context to detect deviations effectively. Larger history sizes may also improve the model's 
# ability to adapt to long-term patterns in the data, making it more reliable for real-time anomaly detection.

    def __init__(self, alpha=0.1, threshold=2.7, history_size=150):
        # Validate initialization parameters
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if threshold <= 0:
            raise ValueError("Threshold must be a positive value.")
        if history_size <= 0:
            raise ValueError("History size must be a positive integer.")

        self.alpha = alpha  # Smoothing factor for EMA
        self.threshold = threshold  # Threshold for anomaly detection
        self.history_size = history_size  # Number of recent deviations to track
        self.ema = None  # Current EMA value (initialized as None)
        self.recent_deviations = []  # List to store recent deviations from EMA

    def update(self, value):
        """Update the EMA with a new value and check for anomalies."""
        
        # Validate input value
        if not isinstance(value, (int, float)):
            raise ValueError("Input value must be a number.")

        # Initialize EMA if it's the first data point
        if self.ema is None:
            self.ema = value
        else:
            # Calculate the new EMA using the formula
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        
        # Calculate the absolute deviation from the current EMA
        deviation = abs(value - self.ema)
        
        # Add this deviation to recent deviations and maintain history size
        self.recent_deviations.append(deviation)
        
        if len(self.recent_deviations) > self.history_size:
            self.recent_deviations.pop(0)  # Remove oldest deviation if exceeding history size

        # Calculate dynamic threshold based on recent deviations
        if len(self.recent_deviations) >= 2:
            mean_deviation = np.mean(self.recent_deviations)  # Mean of recent deviations
            std_deviation = np.std(self.recent_deviations)   # Standard deviation of recent deviations
            
            # Set deviation threshold based on mean and standard deviation
            deviation_threshold = mean_deviation + std_deviation * self.threshold
        else:
            deviation_threshold = self.threshold  # Use static threshold initially
        
        # Return True if the deviation exceeds the dynamic threshold, indicating an anomaly
        return deviation > deviation_threshold

# 3. Simulate a real-time data stream and detect anomalies
def simulate_realtime_stream():
    """Simulate a real-time data stream and visualize detected anomalies."""
    
    data_stream = []  # List to store generated data points
    anomalies = []    # List to store indices of detected anomalies
    
    # Initialize the anomaly detector with specified parameters
    detector = EMAAnomalyDetector(alpha=0.1, threshold=2.7, history_size=150)

    plt.ion()  # Enable interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure for plotting

    for _ in range(1000):  # Simulate generating 1000 data points
        try:
            new_data_point = generate_new_data_point()  # Generate new data point
            
            data_stream.append(new_data_point[0])  # Append new data point to stream

            # Check for anomaly in the newly generated data point
            is_anomaly = detector.update(new_data_point[0])
            if is_anomaly:
                anomalies.append(len(data_stream) - 1)  # Record index of detected anomaly

            # Real-time visualization of data stream and anomalies
            ax.clear()  # Clear previous plot
            ax.plot(data_stream, label='Data Stream', color='blue')  # Plot data stream
            
            ax.scatter(anomalies, np.array(data_stream)[anomalies], color='red', label='Anomalies', zorder=5)  # Mark anomalies
            
            ax.set_title('Real-Time Data Stream with Anomaly Detection')  # Set plot title
            ax.set_xlabel('Time')   # Set x-axis label
            ax.set_ylabel('Value')  # Set y-axis label
            
            ax.legend()             # Show legend on plot
            ax.grid(True)          # Enable grid for better readability

            plt.draw()             # Update plot with new data
            plt.pause(0.1)         # Short pause for real-time effect
            
        except Exception as e:
            print(f"Error during simulation: {e}")  # Print any errors encountered during simulation

    plt.ioff()  # Disable interactive mode after simulation ends
    plt.show()   # Display final plot

# 4. Execute Simulation when script is run directly
if __name__ == "__main__":
    try:
        simulate_realtime_stream()   # Start the real-time simulation function
    except Exception as e:
        print(f"Error during execution: {e}")   # Print any errors encountered during execution