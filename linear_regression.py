import numpy as np
import matplotlib.pyplot as plt

# Our data — problems completed vs test scores
problems = np.array([10, 20, 35, 50, 65, 80])
scores   = np.array([55, 60, 70, 80, 85, 95])

# Step 1 — Start with random m and c
m = 0.0
c = 0.0
learning_rate = 0.0008
epochs = 50000  # how many times we loop

# Step 2 — Training loop
for epoch in range(epochs):
    
    # Make predictions with current m and c
    predictions = m * problems + c
    
    # Calculate loss (MSE)
    loss = np.mean((predictions - scores) ** 2)
    
    # Calculate gradients (direction to adjust m and c)
    dm = np.mean((predictions - scores) * problems)
    dc = np.mean(predictions - scores)
    
    # Update m and c
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.2f}, m = {m:.4f}, c = {c:.4f}")

# Step 3 — See our results
print(f"\nFinal: m = {m:.4f}, c = {c:.4f}")
print(f"Prediction for 45 problems: {m * 45 + c:.1f}")

# Step 4 — Plot
plt.scatter(problems, scores, color='blue', label='Actual data')
plt.plot(problems, m * problems + c, color='red', label='Our line')
plt.xlabel('Problems completed')
plt.ylabel('Test score')
plt.legend()
plt.title('Linear Regression from scratch')
plt.show()

# Plot the loss curve
m_values = np.linspace(-1, 5, 100)
loss_values = [np.mean((m_val * problems + c - scores)**2) 
               for m_val in m_values]

plt.figure()
plt.plot(m_values, loss_values)
plt.xlabel('m value')
plt.ylabel('Loss')
plt.title('Loss curve — how loss changes with m')
plt.axvline(x=m, color='red', linestyle='--', label=f'Final m = {m:.2f}')
plt.legend()
plt.show()