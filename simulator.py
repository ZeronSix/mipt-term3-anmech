import numpy as np
import math
import scipy.integrate

G = 6.6740831E-11
SIMULATION_R_STEP = 0.01
START_DIST_R = 10
INF_DIST_R = 10000

INDEX_ANGLE = 0
INDEX_MIN_DIST = 1
INDEX_TRAJECTORY = 2
INDEX_FULL_ENERGY = 3

class Simulator:
    def __init__(self, planet_radius, func, vel_inf):
        self.density_fn = func
        self.R = planet_radius
        self.M = self.calc_mass(self.R)
        self.K = G * self.M
        self.VEL_INF = vel_inf * math.sqrt(G * self.M / self.R)

        self.ship_pos = np.array([0, 0])
        self.dist = 0
        self.ship_vel = np.array([0, 0])
        self.vel = 0
        self.trajectory = []
        self.full_energy = []
        self.time = 0
        self.shortest_dist = float("inf")

    def calc_mass(self, r_max):
        return scipy.integrate.quad(lambda r: 4 * math.pi * r ** 2 * eval(self.density_fn),
                                    0,
                                    r_max)[0]

    def calc_force(self):
        if self.dist <= self.R:
            return -G * self.calc_mass(self.dist) * self.ship_pos / self.dist ** 3
        else:
            return -G * self.M / self.dist ** 3 * self.ship_pos

    def calc_force_magnitude(self, dist):
        if dist == 0:
            return 0
        if dist <= self.R:
            return G * self.calc_mass(dist) / dist ** 2
        else:
            return G * self.M / dist ** 2

    def calc_potential(self, dist):
        if dist > self.R:
            pot = -self.K / dist
        else:
            pot = -scipy.integrate.quad(lambda r: self.calc_force_magnitude(r),
                                        dist,
                                        self.R)[0] - self.K / self.R
        return pot

    def run(self, impact_param):
        self.shortest_dist = float("inf")
        self.trajectory.clear()
        self.time = 0
        self.full_energy.clear()

        # Angular momentum (areal velocity)
        c = impact_param * self.R * self.VEL_INF

        # Calculate start position
        start_energy = self.VEL_INF ** 2
        ecc = math.sqrt(1 + start_energy * c ** 2 / (self.K ** 2))  # Eccentricity
        inf_angle = np.arccos(-1 / ecc)
        orb_param = c ** 2 / self.K  # Orbital parameter

        start_radius = max(impact_param * self.R, self.R) * START_DIST_R
        start_cos = 1 / ecc * (orb_param / start_radius - 1)
        start_angle = np.arccos(np.clip([start_cos], -1, 1)[0])
        self.ship_pos = np.array([start_radius * np.cos(start_angle), start_radius * np.sin(start_angle)])
        self.dist = self.get_distance()

        # Calculate start velocity
        v = math.sqrt(start_energy + 2 * self.K / start_radius)
        vel_angle = np.pi - np.arcsin(c / start_radius / v)
        vx = v * np.cos(start_angle - vel_angle)
        vy = v * np.sin(start_angle - vel_angle)
        self.ship_vel = np.array([vx, vy])
        self.vel = self.get_velocity()
        while self.get_distance() < 2 * start_radius:
            self.step()

        dot = np.dot(self.ship_vel, np.array([1, 0])) / self.get_velocity()
        return (np.arccos(np.clip(dot, -1, 1)) + inf_angle - np.pi,
                self.shortest_dist,
                self.trajectory.copy(),
                self.full_energy.copy())

    def step(self):
        time_step = SIMULATION_R_STEP * self.R / self.vel
        self.time += time_step
        self.ship_vel += self.calc_force() * time_step
        self.ship_pos += self.ship_vel * time_step
        self.trajectory.append(self.ship_pos.copy())
        self.vel = self.get_velocity()
        self.dist = self.get_distance()

        pot = self.calc_potential(self.dist)
        self.full_energy.append(np.array([self.time,
                                          self.vel ** 2 + 2 * pot]))
        self.shortest_dist = min(self.shortest_dist, self.dist)

    def get_distance(self):
        return np.linalg.norm(self.ship_pos)

    def get_velocity(self):
        return np.linalg.norm(self.ship_vel)
