from RRT import RRT
import numpy as np

class RRTStar(RRT):
    def __init__(self, start, goal, x_limits, y_limits, step_size, max_iter, obstacles, max_steering_angle, max_velocity, wheelbase, rewire_radius, length_fl, width_fl):
        super().__init__(start, goal, x_limits, y_limits, step_size, max_iter, obstacles, max_steering_angle, max_velocity, wheelbase, length_fl, width_fl)
        self.rewire_radius = rewire_radius

    def cost(self, node): # Calculate the cost to reach a node
        cost = 0
        current = node
        while current is not None:
            parent = self.parent[current]
            if parent is not None:
                cost += np.linalg.norm(np.array(current[:2]) - np.array(parent[:2]))
            current = parent
        return cost

    def rewire(self, x_new): # Rewire the tree to optimize paths
        for node in self.tree:
            if np.linalg.norm(np.array(node[:2]) - np.array(x_new[:2])) <= self.rewire_radius:
                if self.is_collision_free(node, x_new):
                    new_cost = self.cost(node) + np.linalg.norm(np.array(node[:2]) - np.array(x_new[:2]))
                    if new_cost < self.cost(x_new):
                        self.parent[x_new] = node

    def plan(self): # Execute RRT*
        for _ in range(self.max_iter):
            x_rand = self.random_sample()
            x_nearest = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand)
            print(_)

            if x_new and self.is_collision_free(x_nearest, x_new):
                self.tree.append(x_new)
                self.parent[x_new] = x_nearest
                self.rewire(x_new)

                # Check if the goal is reachable
                if np.linalg.norm(np.array(x_new[:2]) - np.array(self.goal[:2])) < self.step_size and self.is_collision_free(x_new, self.goal):
                    self.tree.append(self.goal)
                    self.parent[self.goal] = x_new
                    print('path found (not optimal)')
        return self.reconstruct_path()