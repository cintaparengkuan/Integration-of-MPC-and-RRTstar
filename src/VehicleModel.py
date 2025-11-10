import numpy as np

class RoboticForklift:
    def __init__(self):
        # Physical properties
        self.length = 3.35  # meters
        self.width = 1.2  # meters
        self.lifting_height = 3.0  # meters
        self.steering_angle_range = (-np.radians(60), np.radians(60))  # radians
        self.max_velocity = 3.33  # m/s

        # Kinematic model constants
        self.C_alpha_f = 5000  # N/rad
        self.C_alpha_r = 6000  # N/rad
        self.mass = 4000  # kg
        self.moment_of_inertia = 3000  # kg*m^2
        self.l_f = 1.675  # m (center of mass to front axle)
        self.l_r = 1.675  # m (center of mass to rear axle)

        # Initial state
        self.x = 0.0  # X position (m)
        self.y = 0.0  # Y position (m)
        self.psi = 0.0  # Orientation (rad)
        self.v = 0.0  # Lateral velocity (m/s)
        self.r = 0.0  # Yaw rate (rad/s)
        self.delta = 0.0  # Steering angle (rad)
        self.fork_height = 0.0  # Fork height (m)

    def update_steering_angle(self, delta): 
        self.delta = max(self.steering_angle_range[0], min(self.steering_angle_range[1], delta))

    def update_velocity(self, throttle):
        self.v = max(0, min(self.max_velocity, throttle))

    def kinematic_model(self, dt):
        if self.v == 0:  # Avoid division by zero because it kept throwing errors
            return

        # Calculate slip angle
        beta = np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(self.delta))

        # Update position and orientation
        self.x += self.v * np.cos(self.psi + beta) * dt
        self.y += self.v * np.sin(self.psi + beta) * dt
        self.psi += self.r * dt

        # Calculate changes in lateral velocity and yaw rate
        v_dot = (- (self.C_alpha_f + self.C_alpha_r) / (self.mass * self.v)) * self.v \
                + ((-self.v + (self.l_r * self.C_alpha_r - self.l_f * self.C_alpha_f) / (self.mass * self.v)) * self.r) \
                + (self.C_alpha_f / self.mass) * self.delta

        r_dot = ((self.l_r * self.C_alpha_r - self.l_f * self.C_alpha_f) / (self.moment_of_inertia * self.v)) * self.v \
                + ((-self.l_f**2 * self.C_alpha_f - self.l_r**2 * self.C_alpha_r) / (self.moment_of_inertia * self.v)) * self.r \
                + (self.l_f * self.C_alpha_f / self.moment_of_inertia) * self.delta

        self.v += v_dot * dt
        self.r += r_dot * dt

    def move_fork(self, height):
        self.fork_height = max(0, min(self.lifting_height, height))

    def get_state(self):
        return {
            'x': self.x,
            'y': self.y,
            'psi': self.psi,
            'v': self.v,
            'r': self.r,
            'delta': self.delta,
        }

    def get_dimensions(self):
        return self.length, self.width

    def get_corners(self):
        length, width = self.get_dimensions()
        theta = self.psi
        c1 = np.array([self.x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta),
                       self.y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)])
        c2 = np.array([self.x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta),
                       self.y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)])
        c3 = np.array([self.x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta),
                       self.y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)])
        c4 = np.array([self.x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta),
                       self.y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)])
        return c1, c2, c3, c4

