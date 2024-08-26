import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Load data from CSV
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

x_label = data.columns[0]  # First column
y_label = data.columns[1]  # Second column

# Extract x and y values
X = data[x_label].values
Y = data[y_label].values

# Number of data points
N = len(X)

# Calculate necessary sums
sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X ** 2)

# Calculate slope (m) and intercept (c)
m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
c = (sum_y - m * sum_x) / N

# Print the coefficients
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Predict y values
def predict(x):
    return m * x + c

# GUI for user input
def predict_value():
    try:
        # Get the value of X entered by the user
        x_input = float(entry.get())
        # Calculate the predicted Y
        y_output = predict(x_input)
        # Show the result in a message box
        messagebox.showinfo("Prediction Result", f"For X = {x_input}, Predicted Y = {y_output}")
    except ValueError:
        # If the user inputs a non-numeric value
        messagebox.showerror("Invalid Input", "Please enter a valid number.")

# Plotting the data and regression line
def plot_data():
    predicted_Y = predict(X)
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, predicted_Y, color='red', label='Fitted Line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Linear Regression: {x_label} vs {y_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Creating the GUI window
root = tk.Tk()
root.title("Linear Regression Predictor")

root.geometry("300x400")
root.resizable(False, False)  # Disable resizing the window

# Creating an entry widget for user input
label = tk.Label(root, text=f"Enter a value for {x_label}:")
label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=10)

# Creating a button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_value)
predict_button.pack(pady=10)

# Creating a button to plot the data
plot_button = tk.Button(root, text="Plot Data and Line", command=plot_data)
plot_button.pack(pady=10)

# Running the GUI application
root.mainloop()
