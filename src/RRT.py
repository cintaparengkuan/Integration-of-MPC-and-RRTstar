import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, x_limits, y_limits, step_size, max_iter, obstacles, max_steering_angle, max_velocity, wheelbase, length_fl, width_fl):
        self.start = tuple(start)  # (x, y, theta)
        self.goal = tuple(goal)  # (x, y, theta)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.step_size = step_size
        self.max_iter = max_iter
        self.obstacles = obstacles
        self.max_steering_angle = max_steering_angle  # Maximum steering angle in radians
        self.max_velocity = max_velocity  # Maximum velocity in m/s
        self.wheelbase = wheelbase  # Distance between front and rear axle in meters
        self.length_fl = length_fl  # Vehicle length
        self.width_fl = width_fl  # Vehicle width

        # Tree data structure
        self.tree = [self.start]
        self.parent = {self.start: None}

    def random_sample(self): #Generate a random point within the limits
        x = np.random.uniform(self.x_limits[0], self.x_limits[1])
        y = np.random.uniform(self.y_limits[0], self.y_limits[1])
        theta = np.random.uniform(-np.pi, np.pi)
        return (x, y, theta)

    def nearest(self, x_rand): #Find the nearest node in the tree to a random sample
        dists = [np.linalg.norm(np.array(node[:2]) - np.array(x_rand[:2])) for node in self.tree]
        nearest_index = np.argmin(dists)
        return self.tree[nearest_index]

    def steer(self, x_nearest, x_rand): #Steer from the nearest node toward the random sample using non-holonomic constraints.
        x, y, theta = x_nearest
        goal_x, goal_y, goal_theta = x_rand

        # Determine steering angle and velocity
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
        steering_angle = np.clip(angle_to_goal - theta, -self.max_steering_angle, self.max_steering_angle)
        velocity = self.max_velocity

        # Apply the kinematic bicycle model
        dt = 0.1  # Time step for propagation
        for _ in range(int(self.step_size / velocity / dt)):
            x += velocity * np.cos(theta) * dt
            y += velocity * np.sin(theta) * dt
            theta += (velocity / self.wheelbase) * np.tan(steering_angle) * dt

            if not (self.x_limits[0] <= x <= self.x_limits[1] and self.y_limits[0] <= y <= self.y_limits[1]):
                return None

        return (x, y, theta)

    def vehicle_corners(self, x, y, theta):
        half_length = self.length_fl / 2
        half_width = self.width_fl / 2

        c1 = np.array([x + half_length * np.cos(theta) - half_width * np.sin(theta),
                       y + half_length * np.sin(theta) + half_width * np.cos(theta)])
        c2 = np.array([x + half_length * np.cos(theta) + half_width * np.sin(theta),
                       y + half_length * np.sin(theta) - half_width * np.cos(theta)])
        c3 = np.array([x - half_length * np.cos(theta) + half_width * np.sin(theta),
                       y - half_length * np.sin(theta) - half_width * np.cos(theta)])
        c4 = np.array([x - half_length * np.cos(theta) - half_width * np.sin(theta),
                       y - half_length * np.sin(theta) + half_width * np.cos(theta)])

        return [c1, c2, c3, c4]

    def is_collision_free(self, x1, x2):
        for obs in self.obstacles:
            if self.path_intersects_obstacle(x1, x2, obs):
                return False
        return True

    def path_intersects_obstacle(self, p1, p2, obs):
        cx, cy, r = obs
        center = np.array([cx, cy])

        # Linear interpolation to check multiple points along the path
        steps = 10
        for t in np.linspace(0, 1, steps):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            theta = p1[2] + t * (p2[2] - p1[2])
            
            center_car = np.array([x, y])
            corners = self.vehicle_corners(x, y, theta)

            # Check if any corner is inside the obstacle
            for corner in corners:
                if np.linalg.norm(center_car - center) <= r + self.length_fl/2:
                    return True

        return False

    def plan(self):
        for _ in range(self.max_iter):
            x_rand = self.random_sample()
            x_nearest = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand)

            if x_new and self.is_collision_free(x_nearest, x_new):
                self.tree.append(x_new)
                self.parent[x_new] = x_nearest

                # Check if the goal is reachable
                if np.linalg.norm(np.array(x_new[:2]) - np.array(self.goal[:2])) < self.step_size:
                    self.tree.append(self.goal)
                    self.parent[self.goal] = x_new
                    return self.reconstruct_path()
        return None

    def reconstruct_path(self):
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()
        return path

    def plot(self, path=None):
        plt.figure(figsize=(8, 8))
        plt.xlim(self.x_limits)
        plt.ylim(self.y_limits)

        # Obstacles
        for obs in self.obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
            plt.gca().add_artist(circle)

        # Tree
        for node in self.tree:
            if self.parent[node] is not None:
                plt.plot([node[0], self.parent[node][0]], [node[1], self.parent[node][1]], 'g-')

        # Path
        if path:
            px, py = zip(*[(p[0], p[1]) for p in path])
            plt.plot(px, py, 'b-', linewidth=2)

        plt.plot(self.start[0], self.start[1], 'bo', label="Start")
        plt.plot(self.goal[0], self.goal[1], 'go', label="Goal")
        plt.legend()
        plt.grid()
        plt.show()
