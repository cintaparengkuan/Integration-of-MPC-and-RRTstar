import numpy as np
import matplotlib.pyplot as plt

# Robot parameters
robot_radius = 0.5  # Radius of the robot

def velocity_obstacles(robot_pos, robot_vel, obstacles, tau):
    velocity_obstacle_regions = []

    for obs in obstacles:
        obs_pos, obs_vel, shape, size = obs

        # Relative position and velocity
        rel_pos = obs_pos - robot_pos
        rel_vel = robot_vel - obs_vel

        if shape == "circle":
            dist = np.linalg.norm(rel_pos)
            combined_radius = robot_radius + size  # size represents the radius here

            if dist < combined_radius:
                print("Collision imminent! Dist is less than combined radius.")
                continue

            theta = np.arcsin(combined_radius / dist)
            angle_to_obs = np.arctan2(rel_pos[1], rel_pos[0])

            left_bound = angle_to_obs + theta
            right_bound = angle_to_obs - theta

            # Expand VO to account for time horizon
            expansion = rel_pos / tau

            # Store the VO as a tuple: bounds and expansion
            velocity_obstacle_regions.append((left_bound, right_bound, expansion))

        elif shape == "square":
            half_size = size / 2
            vertices = [
                obs_pos + np.array([half_size, half_size]),
                obs_pos + np.array([half_size, -half_size]),
                obs_pos + np.array([-half_size, -half_size]),
                obs_pos + np.array([-half_size, half_size])
            ]

            for vertex in vertices:
                rel_vertex = vertex - robot_pos
                dist = np.linalg.norm(rel_vertex)
                combined_radius = robot_radius

                if dist < combined_radius:
                    print("Collision imminent with square obstacle!")
                    continue

                theta = np.arcsin(combined_radius / dist)
                angle_to_vertex = np.arctan2(rel_vertex[1], rel_vertex[0])

                left_bound = angle_to_vertex + theta
                right_bound = angle_to_vertex - theta

                # Expand VO to account for time horizon
                expansion = rel_vertex / tau

                # Store the VO as a tuple: bounds and expansion
                velocity_obstacle_regions.append((left_bound, right_bound, expansion))

    return velocity_obstacle_regions


def find_collision_free_velocity(robot_vel, preferred_vel, vo_regions, goal_pos, robot_pos):
    samples = 100
    velocities = []
    costs = []

    for angle in np.linspace(0, 2 * np.pi, samples):
        candidate_speed = np.linalg.norm(preferred_vel)
        candidate_vel = np.array([
            candidate_speed * np.cos(angle),
            candidate_speed * np.sin(angle)
        ])

        # Check if candidate velocity is collision-free
        collision_free = True
        for vo in vo_regions:
            left_bound, right_bound, expansion = vo
            candidate_dir = np.arctan2(candidate_vel[1], candidate_vel[0])

            if right_bound <= candidate_dir <= left_bound:
                collision_free = False
                break

        if collision_free:
            velocities.append(candidate_vel)
            # Cost is based on deviation from preferred velocity and angle to the goal
            to_goal = goal_pos - robot_pos
            goal_angle = np.arctan2(to_goal[1], to_goal[0])
            candidate_angle = np.arctan2(candidate_vel[1], candidate_vel[0])
            angle_cost = abs(goal_angle - candidate_angle)
            cost = np.linalg.norm(candidate_vel - preferred_vel) + angle_cost
            costs.append(cost)

    if not velocities:
        print("No collision-free velocities available! Adjusting...")
        # Reduce velocity magnitude if no valid velocity is found
        adjusted_vel = preferred_vel * 0.5
        return adjusted_vel

    # Select the velocity with the minimum cost
    best_idx = np.argmin(costs)
    return velocities[best_idx]

# Just for visualization, remove when used with real environment
def plot_environment(robot_position, robot_velocity, preferred_velocity, new_velocity, obstacles):
    plt.figure()
    plt.quiver(robot_position[0], robot_position[1], robot_velocity[0], robot_velocity[1], color="r", label="Current Velocity")
    plt.quiver(robot_position[0], robot_position[1], new_velocity[0], new_velocity[1], color="g", label="New Velocity")
    plt.quiver(robot_position[0], robot_position[1], preferred_velocity[0], preferred_velocity[1], color="b", label="Preferred Velocity")

    for obs in obstacles:
        obs_pos, _, shape, size = obs

        if shape == "circle":
            circle = plt.Circle(obs_pos, size, color="gray", alpha=0.5, label="Obstacle (Circle)")
            plt.gca().add_artist(circle)

        elif shape == "square":
            half_size = size / 2
            square = plt.Rectangle(
                obs_pos - np.array([half_size, half_size]),
                size, size,
                color="gray", alpha=0.5, label="Obstacle (Square)"
            )
            plt.gca().add_artist(square)

        plt.plot(obs_pos[0], obs_pos[1], 'kx')  # Mark the center of the obstacle

    plt.axis("equal")
    plt.legend()
    plt.title("Environment with Obstacles")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()

# Example usage
robot_position = np.array([0.0, 0.0])
robot_velocity = np.array([0.5, 0.0])
preferred_velocity = np.array([1.0, 0.0])
goal_position = np.array([3.0, 0.0])

# Obstacles defined as [position, velocity, shape, size]
obstacles = [
    [np.array([1.0, 0.0]), np.array([0.0, -0.5]), "circle", 0.3],
    [np.array([2.0, -1.0]), np.array([-0.5, 0.5]), "square", 0.6]
]

time_horizon = 3.0

# Compute velocity obstacles
vo_regions = velocity_obstacles(robot_position, robot_velocity, obstacles, time_horizon)

# Find collision-free velocity
new_velocity = find_collision_free_velocity(robot_velocity, preferred_velocity, vo_regions, goal_position, robot_position)

print("New velocity:", new_velocity)

# Visualization
plot_environment(robot_position, robot_velocity, preferred_velocity, new_velocity, obstacles)