import numpy as np

def plot_ode(t, x_0, x_t, title, filename="ode_trajectories.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i in range(x_0.shape[0]):
        plt.plot(t, x_t[i, :], label=f'Trajectory {i+1}')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('State')
    # plt.legend()
    plt.grid()
    plt.savefig(filename)

def linear_ode():
    x_0 = np.linspace(-10, 10, 20)
    t = np.linspace(0, 1, 200)
    theta = 10

    x_t = []
    for time in t:
        x_t.append(np.exp(-theta * time) * x_0)
    
    x_t = np.array(x_t)
    x_t = x_t.T  # Transpose for easier plotting
    return x_t, t, x_0

def linear_ode_euler(steps=200):
    x_0 = np.linspace(-10, 10, 20)
    t = np.linspace(0, 1, steps)
    delta_t = 1 / steps
    theta = 10

    x_t = []
    x_curr = x_0.copy()
    for time in t:
        x_t.append(x_curr)
        x_curr = x_curr + delta_t * -theta * x_curr
    
    x_t = np.array(x_t)
    x_t = x_t.T  # Transpose for easier plotting
    return x_t, t, x_0

def linear_sde_euler(steps=200):
    x_0 = np.linspace(-10, 10, 20)
    t = np.linspace(0, 1, steps)
    delta_t = 1 / steps
    theta = 10
    sigma = 1

    x_t = []
    x_curr = x_0.copy()
    for time in t:
        epsilon = np.random.normal(0, 1, size=x_0.shape)
        x_t_h = x_curr + delta_t * (-theta * x_curr) + np.sqrt(delta_t) * sigma * epsilon
        x_t.append(x_t_h)
        x_curr = x_t_h
    
    x_t = np.array(x_t)
    x_t = x_t.T  # Transpose for easier plotting
    return x_t, t, x_0

if __name__ == "__main__":
    x_t, t, x_0 = linear_ode()
    plot_ode(t, x_0, x_t, title="Linear ODE Trajectories", filename="content/posts/flow/linear_ode_trajectories.png")

    x_t_euler, t_euler, x_0_euler = linear_ode_euler()
    plot_ode(t_euler, x_0_euler, x_t_euler, title="Linear ODE Euler Trajectories", filename="content/posts/flow/linear_ode_euler_trajectories.png")

    x_t_sde, t_sde, x_0_sde = linear_sde_euler()
    plot_ode(t_sde, x_0_sde, x_t_sde, title="Linear SDE Euler Trajectories", filename="content/posts/flow/linear_sde_euler_trajectories.png")