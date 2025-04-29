import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Contour Plotting

def f(x1, x2):
    return (x2 - x1**2)**2 + (1 - x1)**2

# Create grid
x1 = np.linspace(-6, 6, 400)
x2 = np.linspace(-6, 6, 400)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate the function on the grid
Z = f(X1, X2)

# Make the contour plot
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=[
    0, 0.10, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300, 400
])
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(1, 1, 'ro', label='Minimum (1,1)')
plt.title(r'Contour plot of $f(x_1, x_2)$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.grid(True)
plt.show()

def f(x):  # Function
    return (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):  # Gradient of our function
    df_dx1 = -4 * (x[1] - x[0]**2) * x[0] + 2 * (x[0] - 1)
    df_dx2 = 2 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def armijo(x, p, grad, alpha=2, c=0.5, rho=0.95):  # Perform Armijo backtracking line search with specified conditions.
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad, p):
        alpha = rho * alpha
    return alpha

# Gradient Descent (Steepest Descent)
def gradient_descent(x0):
    x = x0.copy()
    grad = grad_f(x)
    epsilon = 1e-3
    count = 0
    path = [x.copy()]  # store path

    while np.linalg.norm(grad) >= epsilon:
        print(f"Iteration {count}: x = {x}, f(x) = {f(x)}, grad_f(x) = {grad}")
        p = -grad  # Steepest descent direction
        alpha = armijo(x, p, grad)  # Perform line search using Armijo
        x = x + alpha * p  # Iterate
        grad = grad_f(x)
        path.append(x.copy())  # store path
        count += 1

    print(f"Iteration {count}: x = {x}, f(x) = {f(x)}, grad_f(x) = {grad}")
    return x, count, f(x), np.array(path)

# Method of choice (DFP)
def DFP(x0):
    x = x0.copy()
    H = np.eye(2)  # Initialize H_0 as identity matrix
    grad = grad_f(x)
    epsilon = 1e-3
    count = 0
    path = [x.copy()]  # store path

    while np.linalg.norm(grad) >= epsilon:  # Loop until given stopping conditions
        print(f"Iteration {count}: x = {x}, f(x) = {f(x)}, grad_f(x) = {grad}")
        p = -H @ grad  # Set p_k = -H_k * grad_f(x_k)
        alpha = armijo(x, p, grad)  # Perform line search using Armijo
        x_new = x + alpha * p  # Iterate
        grad_new = grad_f(x_new)

        deltaxk = x_new - x
        deltagk = grad_new - grad

        if np.dot(deltaxk, deltagk) != 0:
            # Update H_{k+1} using DFP formula
            H = H + np.outer(deltaxk, deltaxk) / np.dot(deltaxk, deltagk) - (H @ np.outer(deltagk, deltagk) @ H) / (deltagk @ H @ deltagk)

        x = x_new
        grad = grad_new
        path.append(x.copy())  # store path
        count += 1

    print(f"Iteration {count}: x = {x}, f(x) = {f(x)}, grad_f(x) = {grad}")
    return x, count, f(x), np.array(path)

def final_results(result, iterations, f_value):
    print("Minimum at:", result)
    print("Function value:", f_value)
    print("Iterations:", iterations)

def animate_paths(paths, labels, colors, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    contours = ax.contour(X1, X2, Z, levels=[
        0, 0.10, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300, 400
    ])
    ax.clabel(contours, inline=True, fontsize=8)

    min_dot, = ax.plot(1, 1, 'ro', label='Minimum (1,1)')

    ax.set_title(title)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True)

    lines = []
    points = []

    # Create line and point objects, one for each path/method
    for label, color in zip(labels, colors):
        line, = ax.plot([], [], color=color, lw=2, label=label)  # line for path
        point, = ax.plot([], [], 'o', color=color, markersize=2)  # moving point
        lines.append(line)
        points.append(point)

    ax.legend()

    # Initialization function for the animation:
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points

    # Update function called on each frame of the animation
    def update(frame):
        for i, path in enumerate(paths):
            if frame < len(path):
                # Update the path line (all points up to this frame)
                lines[i].set_data(path[:frame + 1, 0], path[:frame + 1, 1])
                # Update the current point (single point at this frame)
                points[i].set_data([path[frame, 0]], [path[frame, 1]])
        return lines + points

    # Use the longest path length as the total number of animation frames
    max_len = max(len(path) for path in paths)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=max_len,
        init_func=init, interval=100, blit=True, repeat=False
    )

    # Save the animation to an MP4 file using ffmpeg
    ani.save(filename, writer='ffmpeg', fps=2)

    # Show the final frame
    plt.show()

# Results

# Results for (-4, 4)
print("x0=(-4,4)")
x0 = np.array([-4.0, 4.0])
print("Gradient Descent Method:")
result, iterations, f_value, path_gd = gradient_descent(x0)
final_results(result, iterations, f_value)
print("DFP Method:")
result, iterations, f_value, path_dfp = DFP(x0)
final_results(result, iterations, f_value)

animate_paths([path_gd, path_dfp],
              labels=["Gradient Descent", "DFP"],
              colors=["blue", "green"],
              title="Animated Paths from x0 = (-4,4)",
              filename="animation_minus4_4.mp4")

# Results for (-4, -4)
print("x0=(-4,-4)")
x0 = np.array([-4.0, -4.0])
print("Gradient Descent Method:")
result, iterations, f_value, path_gd = gradient_descent(x0)
final_results(result, iterations, f_value)
print("DFP Method:")
result, iterations, f_value, path_dfp = DFP(x0)
final_results(result, iterations, f_value)

animate_paths([path_gd, path_dfp],
              labels=["Gradient Descent", "DFP"],
              colors=["blue", "green"],
              title="Animated Paths from x0 = (-4,-4)",
              filename="animation_minus4_minus4.mp4")
