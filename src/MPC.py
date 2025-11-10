import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import random

N = 10
Width_FL = 1.5
Length_FL = 3


def current_state_np(vehicle, x, y, theta):
    #corners forklift
    c1 = np.array([x + 1/2 * Length_FL * np.cos(theta) + 1/2 * Width_FL * np.sin(theta),
                   y + 1/2 * Length_FL * np.sin(theta) - 1/2 * Width_FL * np.cos(theta)])
    c2 = np.array([x + 1/2 * Length_FL * np.cos(theta) - 1/2 * Width_FL * np.sin(theta),
                   y + 1/2 * Length_FL * np.sin(theta) + 1/2 * Width_FL * np.cos(theta)])
    c3 = np.array([x - 1/2 * Length_FL * np.cos(theta) - 1/2 * Width_FL * np.sin(theta),
                   y - 1/2 * Length_FL * np.sin(theta) + 1/2 * Width_FL * np.cos(theta)])
    c4 = np.array([x - 1/2 * Length_FL * np.cos(theta) + 1/2 * Width_FL * np.sin(theta),
                   y - 1/2 * Length_FL * np.sin(theta) - 1/2 * Width_FL * np.cos(theta)])
    return c1, c2, c3, c4


#Function that creates linear moving obstacles
def create_lin_moving_obstacle(pos_init, pos_end, vel, radi):
    x_init = pos_init[0]
    y_init = pos_init[1]
    
    x_end = pos_end[0]
    y_end = pos_end[1]
    
    dist = np.sqrt((x_end - x_init) ** 2 + (y_end - y_init) ** 2)
    numbers = list(range(1, 40))  # List of numbers from 0 to 200
    time = dist/vel
    start_int = random.choice(numbers)  # Randomly pick one
    start_int = 1
    
    x = []
    y = []
    radius = []
    for i in range(start_int):
        x.append(x_init)
        y.append(y_init)
        radius.append(radi)
        
    for i in range(int(time)):
        x.append(x[-1] + (x_end - x[-1]) / time)
        y.append(y[-1] + (y_end - y[-1]) / time)
        radius.append(radi)
    
    
    for i in range(1000):
        x.append(x[-1])
        y.append(y[-1])
        radius.append(radi)
        

    return np.array([x[:1001], y[:1001], radius[:1001]])


#Function that creates moving obstacles that move in a sinusoidal manner
def create_sin_moving_obstacle(pos_init, pos_end, vel, radi, amp):
    Width_FL = 1.5
    Length_FL = 3
    
    x_init = pos_init[0]
    y_init = pos_init[1]
    
    x_end = pos_end[0]
    y_end = pos_end[1]
    
    dist = np.sqrt((x_end - x_init) ** 2 + (y_end - y_init) ** 2)
    numbers = list(range(1, 40))  # List of numbers from 0 to 200
    time = dist/vel
    start_int = random.choice(numbers)  # Randomly pick one
    start_int = 1
    
    x = []
    y = []
    radius = []
    for i in range(start_int):
        x.append(x_init)
        y.append(y_init)
        radius.append(radi)
        
    for i in range(int(time)):
        t = i / time 
        x_curr = x_init + t * (x_end - x_init)
        y_curr = y_init + t * (y_end - y_init) + amp * np.sin(2 * np.pi * t * 10) # sine wave pattern 
        x.append(x_curr) 
        y.append(y_curr)
        radius.append(radi)
    
    
    for i in range(1000):
        x.append(x[-1])
        y.append(y[-1])
        radius.append(radi)
        

    return np.array([x[:1001], y[:1001], radius[:1001]])


#pos_init and pos_target are both 4 dimensional vectors in which:
#pos_init or pos_target = [x_pos, y_pos, velocity, theta]
#path is defined by: [x_int1, y_int1, x_int2, y_int2]
def mpc_control(vehicle, N, pos_init, pos_target, path, obstacles, moving_obstacles, timestamp, cost_angle):
    dt = 0.2        #Timestep
    L = Length_FL   #Length vehicle
    
    #Initial values
    x_init = pos_init[0]        
    y_init = pos_init[1]
    vel_init = pos_init[2]
    theta_init = pos_init[3]

    #Reference or goal values
    x_ref = pos_target[0]
    y_ref = pos_target[1]
    theta_ref = pos_target[3]
    
    #Reference values of RRT* path (first next point on path and second next point on path)
    x_int1 = path[0]
    y_int1 = path[1]
    x_int2 = path[2]
    y_int2 = path[3]
    
    
    #Weights for the cost function
    weight_endpointx = 0        #This value is > 0 if only MPC is used without RRT*
    weight_endpointy = 0        #This value is > 0 if only MPC is used without RRT*
    weight_intpoint1 = 500      #Weight for minimising distance to next point on RRT*
    weight_intpoint2 = 900      #Weight for minimising distance to second next point on RRT*
    angle_cost = cost_angle     #Weight for right orientation at endposition                 
    
    #Initialisation of cost and constraints
    cost = 0.
    
    #variable definition of parameters x, y, v, a, steering_angle, theta
    #in principle all these values can be changed for optimisation, because of the constraints
    #defined later only the steering_angle and acceleration variables can be changed for optimisation.
    x = ca.MX.sym('x', N + 1)
    y = ca.MX.sym('y', N + 1)
    v = ca.MX.sym('v', N + 1)
    a = ca.MX.sym('a', N)
    steering_angle = ca.MX.sym('steering_angle', N)
    theta = ca.MX.sym('theta', N+1)
    vars = ca.vertcat(x, y, v, a, steering_angle, theta)
    
    
    #List constraints
    g = []
    
    #Add initial constraints 
    g.append(x[0] - x_init)
    g.append(y[0] - y_init)
    g.append(v[0] - vel_init)
    g.append(theta[0] - theta_init)
    
    #Loop for amount of timesteps MPC looks into the future
    for k in range(N): 
        
        #Add costfunctions
        cost += weight_endpointx * (x_ref - x[k]) ** 2
        cost += weight_endpointy * (y_ref - y[k]) ** 2
        cost += weight_intpoint1 * (x_int1 - x[k]) ** 2
        cost += weight_intpoint1 * (y_int1 - y[k]) ** 2
        cost += weight_intpoint2 * (x_int2 - x[k]) ** 2
        cost += weight_intpoint2 * (y_int2 - y[k]) ** 2
        cost += angle_cost * (theta_ref - theta[k]) ** 2
        
        #Add vehicle dynamics constraints
        g.append(x[k+1] - (x[k] + v[k] * ca.cos(theta[k]) * dt))
        g.append(y[k+1] - (y[k] + v[k] * ca.sin(theta[k]) * dt))
        g.append(theta[k+1] - (theta[k] + (v[k] * ca.tan(steering_angle[k]) / L) * dt))
        g.append(v[k+1] - (v[k] + a[k] * dt))
        
        #Add velocity acceleration and steering_angle constraints
        g.append(a[k])
        g.append(steering_angle[k])
        g.append(v[k])
        
        #Add constraints for all the defined static obstacles
        for obs in obstacles:
            g.append(ca.sqrt((x[k] - obs[0])**2 + (y[k] - obs[1]) ** 2))
  
        #Add constraints for all the defined moving obstacles
        for obs in moving_obstacles:
            g.append(ca.sqrt((x[k] - obs[0][timestamp + k]) ** 2 + (y[k] - obs[1][timestamp + k]) ** 2))

    x0 = np.zeros(vars.shape[0])
    x0[:N+1] = x_init               # Initialize x positions with current x
    x0[N+1:2*(N+1)] = y_init        # Initialize y positions with current y
    x0[2*(N+1):3*(N+1)] = vel_init  # Initialize velocities
    x0[5*(N+1):] = theta_init       # Initialize orientations
     
    #define initial bounds
    lbg = [0.0] * 4      
    ubg = [0.0] * 4


    for k in range(N):
        #general bounds for formulas
        lbg.extend([0.0] * 4)
        ubg.extend([0.0] * 4)
        
        #bound for acceleration
        lbg.extend([-2])
        ubg.extend([2])
        
        #bound for steering_angle
        lbg.extend([-np.pi/3])
        ubg.extend([np.pi/3])
        
        
        #bound for velocity
        lbg.extend([-3.3])
        ubg.extend([3.3])
        
        for obs in obstacles:
            #bound for static obstacles
            lbg.extend([obs[2] + Width_FL/2 + 0.5])
            ubg.extend([1000000])
        
        for obs in moving_obstacles:
            #bound for moving obstacles
            lbg.extend([obs[2][0] + Length_FL/2 + 0.8])
            ubg.extend([1000000])
    
    nlp = {'x': vars, 'f': cost, 'g': ca.vertcat(*g)}  # vars: decision variables, f: cost, g: constraints

    # Solves the problem
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    solution = solver(x0=x0, lbg=lbg, ubg=ubg)

    x = solution['x']
    # We return the MPC input and the next state (and also the plan for visualization)
    return x


import matplotlib.animation as animation

#Function for animating result
def animate_simulation(x, y, theta, vel, obs, mov_obs, target, corner_positions, path_x, path_y, name_ani):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title("MPC Simulation with Moving Obstacles")
    
    # Plot the target point
    target_plot, = ax.plot(target[0], target[1], 'r*', markersize=10, label="Target")

    path_plot, = ax.plot(path_x, path_y, label = "RRT_star")

    # Plot static obstacles
    static_obstacle_patches = []
    for i, obstacle in enumerate(obs):
        static_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color="gray", alpha=0.5, label="Static Obstacle" if i == 0 else None)
        static_obstacle_patches.append(static_patch)
        ax.add_patch(static_patch)

    # Initialize moving obstacles
    moving_obstacle_patches = []
    for i, mov_ob in enumerate(mov_obs):
        moving_patch = plt.Circle((mov_ob[0][0], mov_ob[1][0]), mov_ob[2][0], color="blue", alpha=0.5, label="Moving Obstacle" if i == 0 else None)
        moving_obstacle_patches.append(moving_patch)
        ax.add_patch(moving_patch)

    # Initialize vehicle position and corners
    vehicle_path, = ax.plot([], [], 'g-', linewidth=2, label="Vehicle Path")
    vehicle_patch, = ax.plot([], [], 'g-', linewidth=1, label="Vehicle Body")
    
    def update(frame):
        # Update vehicle path
        vehicle_path.set_data(x[:frame], y[:frame])

        # Update vehicle corners
        if frame < len(corner_positions):
            corners = corner_positions[frame]
            c1, c2, c3, c4 = corners
            corner_x = [c1[0], c2[0], c3[0], c4[0], c1[0]]
            corner_y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
            vehicle_patch.set_data(corner_x, corner_y)

        # Update moving obstacles
        for i, mov_ob in enumerate(mov_obs):
            if frame < len(mov_ob[0]):  # Ensure frame is within range
                moving_obstacle_patches[i].center = (mov_ob[0][frame], mov_ob[1][frame])
        
        return vehicle_path, vehicle_patch, moving_obstacle_patches

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=200, blit=False, repeat=False)

    # Add legend
    ax.legend()

    # Save animation
    ani.save(name_ani, writer='pillow')
    
    # Display animation
    plt.show()

    return ani