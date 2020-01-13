import pygame 
import numpy as np 
import random 
import time 
import os
import math
import matplotlib.pyplot as plt 
import matplotlib.style as style
from scipy.spatial.distance import euclidean

class Robot(pygame.sprite.Sprite):
    def __init__(self, pose=[0,0,0]):
        super(Robot, self).__init__()
        self.pose = pose
        self.robot_image = pygame.image.load("../images/robot.png").convert_alpha()
        self.size = self.robot_image.get_size()
        self.robot_image = pygame.transform.scale(self.robot_image, (int(self.size[0]*0.075), int(self.size[1]*0.075)))
        self.image = pygame.transform.rotate(self.robot_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0],self.pose[1])

    def move_forward(self, dist, stddev_m = 0.1, stddev_a = 0.01):
        x = self.pose[0]
        y = self.pose[1]
        init_theta = self.pose[2]
        theta = init_theta
        if stddev_a > 0:
            theta = random.gauss(init_theta, stddev_a)
        if stddev_m > 0:
            dist = random.gauss(dist, stddev_m*100)
        x = x + dist * np.cos(theta)
        y = y - dist * np.sin(theta) 
        self.pose = [x, y, theta]
        self.image = pygame.transform.rotate(self.robot_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0],self.pose[1])

    def turn(self, d_theta, stddev_a = 0.01):
        x = self.pose[0]
        y = self.pose[1]
        init_theta = self.pose[2]
        if stddev_a > 0:
            d_theta = random.gauss(d_theta, stddev_a)
        theta = self.normalize_angle(init_theta + d_theta)
        self.pose = [x, y, theta]
        self.image = pygame.transform.rotate(self.robot_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0],self.pose[1])

    def normalize_angle(self, angle):
        while angle >= 2.0 * np.pi:
            angle = angle - 2.0 * np.pi
        while angle < 0:
            angle = angle + 2.0 * np.pi
        return angle

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, pose=[0,0,0]):
        super(Obstacle, self).__init__()
        self.pose = pose
        self.color = (84, 3, 100)
        self.image = pygame.image.load("../images/wall.png").convert_alpha()
        self.size = self.image.get_size()
        self.image = pygame.transform.scale(self.image, (int(self.size[0]*1.25), int(self.size[1]*0.25)))
        self.image = pygame.transform.rotate(self.image, np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0],self.pose[1])

class Block(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 0

class Particle(pygame.sprite.Sprite):
    def __init__(self, pose=[0,0,0], width=15, height=10, color=(252, 3, 3), probability=1.0, occupancy_grid=5):
        super(Particle, self).__init__()
        self.pose = pose
        self.width = width
        self.height = height
        self.probability = probability
        self.occupancy_grid = occupancy_grid
        self.color = (color[0], color[1], color[2])
        self.particle_image = pygame.Surface([width, height], pygame.SRCALPHA, 32)
        self.particle_image = self.particle_image.convert_alpha()
        self.draw_particle()
        self.image = pygame.transform.rotate(self.particle_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0]+900,self.pose[1])

    def get_grid_pos(self, grid_size):
        grid_x = int(self.pose[0] // grid_size)
        grid_y = int(self.pose[1] // grid_size)
        return grid_x, grid_y

    def draw_particle(self):
        p0 = np.array([[0], [0]])
        p1 = np.array([[(self.width)], [(self.height//2)]])
        p2 = np.array([[0], [self.height]])
        if self.probability < 0.5:
            alpha = 0.5 * 255
        else:
            alpha = self.probability*255 
        # print("alpha ", int(alpha))
        pygame.draw.polygon(self.particle_image, (self.color[0], self.color[1], self.color[2], int(alpha)), [p0, p1, p2])

    def redraw(self):
        # self.pose = [x, y, theta]
        self.draw_particle()
        self.image = pygame.transform.rotate(self.particle_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0]+900,self.pose[1])

    def move_forward(self, dist, stddev_m = 0.1, stddev_a = 0.01):
        x = self.pose[0]
        y = self.pose[1]
        init_theta = self.pose[2]
        theta = init_theta
        if stddev_a > 0:
            theta = random.gauss(init_theta, stddev_a)
        if stddev_m > 0:
            dist = random.gauss(dist, stddev_m*100)
        x = x + dist * np.cos(theta)
        y = y - dist * np.sin(theta) 
        self.pose = [x, y, theta]
        self.draw_particle()
        self.image = pygame.transform.rotate(self.particle_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0]+900,self.pose[1])
    
    def turn(self, d_theta, stddev_a = 0.01):
        x = self.pose[0]
        y = self.pose[1]
        init_theta = self.pose[2]
        if stddev_a > 0:
            d_theta = random.gauss(d_theta, stddev_a)
        theta = self.normalize_angle(init_theta + d_theta)
        self.pose = [x, y, theta]
        self.draw_particle()
        self.image = pygame.transform.rotate(self.particle_image.copy(), np.degrees(self.pose[2]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.pose[0]+900,self.pose[1])

    def normalize_angle(self, angle):
        while angle >= 2.0 * np.pi:
            angle = angle - 2.0 * np.pi
        while angle < 0:
            angle = angle + 2.0 * np.pi
        return angle

class OccupancyGridMap:
    def __init__(self, map_width, map_height, grid_size=10):
        self.map_width = map_width
        self.map_height = map_height 
        self.grid_size = grid_size
        self.data = np.zeros((self.map_height//self.grid_size, self.map_width//self.grid_size), dtype=np.int)

class ObstacleRunEnv:
    def __init__(self, n_obstacles=10, n_particles=10, occupancy_grid=10):
        self.scale = 100
        self.map_width = 900
        self.map_height = 600
        self.screen_width = self.map_width * 2 
        self.screen_height = self.map_height
        self.n_obstacles = n_obstacles
        self.n_particles = n_particles
        pygame.init()
        pygame.display.set_caption("Simultaneous Localization and Mapping")
        self.window_size = [self.screen_width, self.screen_height]
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

        self.field_color = (50, 168, 52)
        self.vision_range = np.radians(40)
        self.vision_color = (52, 110, 235)
        self.collide_color = (242, 2, 14)
        self.occupancy_color = (122, 119, 119)
        self.free_color = (255,255,255)
        self.obstacle_color = (0, 0, 0)
        self.particle_color = (252, 3, 3)

        self.screen.fill(self.field_color, pygame.Rect(0,0,self.map_width,self.map_height))
        self.obstacles_sprites = pygame.sprite.Group()
        self.robot_sprites = pygame.sprite.Group()
        self.place_random_obstacles()
        self.place_robot()
        self.obstacles_sprites.draw(self.screen)
        self.occupancy_grid = occupancy_grid
        self.map = OccupancyGridMap(self.map_width, self.map_height, grid_size=self.occupancy_grid)
        self.fill_map_data()
        self.particles_sprites = pygame.sprite.Group()
        # self.estimates = Particle(color=(152, 3, 3))
        x, y, theta = self.get_random_pose()
        self.estimates = Particle(pose=[x,y,theta], width=45, height=30, color=(86, 235, 52), probability=1)
        self.particles_sprites.add(self.estimates)
        # self.n_particles = 10
        self.generate_random_particles()

    def generate_random_particles(self):
        self.particles = []
        for i in range(self.n_particles):
            x, y, theta = self.get_random_pose()
            if i == 0:
                x = self.robot.pose[0]
                y = self.robot.pose[1]
                theta = self.robot.pose[2]
            particle = Particle(pose=[x,y,theta], color=self.particle_color, probability=1)
            self.particles.append(particle)

        for i in range(self.n_particles):    
            self.particles_sprites.add(self.particles[i])

    def fill_map_data(self):
        for i in range(0, self.map_width, self.occupancy_grid):
            for j in range(0, self.map_height, self.occupancy_grid):
                block = Block((0,0,0), self.occupancy_grid, self.occupancy_grid)
                block.rect.x = i 
                block.rect.y = j
                collide = 1
                xx, yy = block.rect.bottomright
                if xx >=0 and xx < self.map_width and yy >= 0 and yy < self.map_height:
                    tl_r, tl_g, tl_b, _ = self.screen.get_at(block.rect.topleft)
                    tr_r, tr_g, tr_b, _ = self.screen.get_at(block.rect.topright)
                    bl_r, bl_g, bl_b, _ = self.screen.get_at(block.rect.bottomleft)
                    br_r, br_g, br_b, _ = self.screen.get_at(block.rect.bottomright)
                    if (tl_r, tl_g, tl_b) == self.field_color and (tr_r, tr_g, tr_b) == self.field_color \
                        and (bl_r, bl_g, bl_b) == self.field_color and (br_r, br_g, br_b) == self.field_color:
                            collide = 0
                else:
                    collide = 0
                self.map.data[j//self.occupancy_grid, i//self.occupancy_grid] = collide
        # di tepi semua obstacle
        self.map.data[0,:] = 1
        self.map.data[:,0] = 1
        self.map.data[-1,:] = 1
        self.map.data[:,-1] = 1

    def draw_map(self):
        h, w = self.map.data.shape
        for i in range(0, w):
            for j in range(0, h):
                if self.map.data[j,i] == 0:
                    self.screen.fill(self.free_color, pygame.Rect(self.map_width+i*self.occupancy_grid,j*self.occupancy_grid,self.occupancy_grid,self.occupancy_grid))
                else:
                    self.screen.fill(self.obstacle_color, pygame.Rect(self.map_width+i*self.occupancy_grid,j*self.occupancy_grid,self.occupancy_grid,self.occupancy_grid))

    def place_random_obstacles(self):
        self.obstacles = []
        for i in range(self.n_obstacles):
            if i > 0:
                acceptable = False
                while not acceptable:
                    x, y, theta = self.get_random_pose()
                    obstacle = Obstacle(pose=[x,y,theta])
                    collide = False
                    # Make sure obstacle inside map
                    tl_x, tl_y = obstacle.rect.topleft 
                    tr_x, tr_y = obstacle.rect.topright 
                    bl_x, bl_y = obstacle.rect.bottomleft 
                    br_x, br_y = obstacle.rect.bottomright 
                    if tl_x >= 0 and tl_x < self.map_width and \
                        tl_y >= 0 and tl_y < self.map_height and \
                        tr_x >= 0 and tr_x < self.map_width and \
                        tr_y >= 0 and tr_y < self.map_height and \
                        bl_x >= 0 and bl_x < self.map_width and \
                        bl_y >= 0 and bl_y < self.map_height and \
                        br_x >= 0 and br_x < self.map_width and \
                        br_y >= 0 and br_y < self.map_height:
                            collide = False
                    else:
                        collide = True

                    for j in range(i):
                        if pygame.sprite.collide_rect(obstacle, self.obstacles[j]):
                            collide = True
                            break
                    if not collide:
                        self.obstacles.append(obstacle)
                        acceptable = True
            else:
                x, y, theta = self.get_random_pose()
                obstacle = Obstacle(pose=[x,y,theta])
                acceptable = False
                while not acceptable:
                    x, y, theta = self.get_random_pose()
                    obstacle = Obstacle(pose=[x,y,theta])
                    collide = False
                    # Make sure obstacle inside map
                    tl_x, tl_y = obstacle.rect.topleft 
                    tr_x, tr_y = obstacle.rect.topright 
                    bl_x, bl_y = obstacle.rect.bottomleft 
                    br_x, br_y = obstacle.rect.bottomright 
                    if tl_x >= 0 and tl_x < self.map_width and \
                        tl_y >= 0 and tl_y < self.map_height and \
                        tr_x >= 0 and tr_x < self.map_width and \
                        tr_y >= 0 and tr_y < self.map_height and \
                        bl_x >= 0 and bl_x < self.map_width and \
                        bl_y >= 0 and bl_y < self.map_height and \
                        br_x >= 0 and br_x < self.map_width and \
                        br_y >= 0 and br_y < self.map_height:
                            collide = False
                    else:
                        collide = True
                    if not collide:
                        self.obstacles.append(obstacle)
                        acceptable = True

        for i in range(self.n_obstacles):
            self.obstacles_sprites.add(self.obstacles[i])

    def place_robot(self):
        acceptable = False
        while not acceptable:
            # x, y, theta = self.get_random_pose()
            x = self.map_width // 2
            y = self.map_height // 2
            theta = np.radians(random.randint(0,360))
            self.robot = Robot(pose=[x,y,theta])
            collide = False
            # Make sure robot inside map
            tl_x, tl_y = self.robot.rect.topleft 
            tr_x, tr_y = self.robot.rect.topright 
            bl_x, bl_y = self.robot.rect.bottomleft 
            br_x, br_y = self.robot.rect.bottomright 
            if tl_x >= 0 and tl_x < self.map_width and \
                tl_y >= 0 and tl_y < self.map_height and \
                tr_x >= 0 and tr_x < self.map_width and \
                tr_y >= 0 and tr_y < self.map_height and \
                bl_x >= 0 and bl_x < self.map_width and \
                bl_y >= 0 and bl_y < self.map_height and \
                br_x >= 0 and br_x < self.map_width and \
                br_y >= 0 and br_y < self.map_height:
                    collide = False
            else:
                collide = True
            for j in range(self.n_obstacles):
                if pygame.sprite.collide_rect(self.robot, self.obstacles[j]):
                    collide = True
                    break
            if not collide:
                acceptable = True
        self.robot_sprites.add(self.robot)

    def get_random_pose(self):
        x = random.randint(0,self.map_width)
        y = random.randint(0,self.map_height)
        theta = np.radians(random.randint(0, 360))
        return x, y, theta

    def particles_move_forward(self, dist, stddev_m = 0.1, stddev_a = 0.01):
        for i in range(self.n_particles):
            self.particles[i].move_forward(dist, stddev_m=stddev_m, stddev_a=stddev_a)

    def particles_turn(self, d_theta, stddev_a = 0.01):
        for i in range(self.n_particles):
            self.particles[i].turn(d_theta, stddev_a=stddev_a)

    def check_sensor(self, min_length=50, max_length=300):
        distances = []
        scan_angles = []
        obstacles = []
        for angle_range in range(-40,50,10):
            r = min_length
            obstacle = 0
            for dist_range in range(min_length, max_length, 10):
                x = self.robot.pose[0]
                y = self.robot.pose[1]
                theta = -self.robot.pose[2]
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
                dy = dist_range*np.sin(np.radians(angle_range))
                dx = dist_range*np.cos(np.radians(angle_range))
                p1 = R.dot(np.array([[dx], [-dy]]))
                scan_point = (int(p1[0,0]+x), int(p1[1,0]+y))
                r = dist_range
                r_grid = r // self.occupancy_grid
                if scan_point[0] >= 0 and scan_point[0] < self.map_width and scan_point[1] >= 0 and scan_point[1] < self.map_height:
                    rr, gg, bb, _ = self.screen.get_at(scan_point)
                    if (rr, gg, bb) == self.field_color:
                        obstacle = 0
                        pygame.draw.circle(self.screen, self.collide_color, scan_point, 3)
                    else:
                        obstacle = 1
                        break
                else:
                    obstacle = 1
                    break
            distances.append(r_grid)
            scan_angles.append(angle_range)
            obstacles.append(obstacle)
        return distances, scan_angles, obstacles

    def get_correlation(self, distances, scan_angles, obstacles):
        correlations = []
        for i in range(self.n_particles):
            px, py = self.particles[i].get_grid_pos(self.occupancy_grid)
            particle_obstacles = []
            particle_heading = -self.particles[i].pose[2]
            R = np.array([[np.cos(particle_heading), -np.sin(particle_heading)],
                          [np.sin(particle_heading), np.cos(particle_heading)]])
            for j in range(len(obstacles)):
                r = distances[j]
                scan_angle = scan_angles[j]
                dy = r*np.sin(np.radians(scan_angle))
                dx = r*np.cos(np.radians(scan_angle))
                p1 = R.dot(np.array([[dx], [-dy]]))
                cell_x = int(p1[0,0]+px)
                cell_y = int(p1[1,0]+py)
                if cell_x >= 0 and cell_x < self.map.data.shape[1] and cell_y >= 0 and cell_y < self.map.data.shape[0]:
                    cell_data = self.map.data[cell_y, cell_x]
                else:
                    cell_data = 1
                particle_obstacles.append(cell_data)
            # print(i, particle_obstacles)
            total_equal = np.sum(np.equal(np.array(obstacles), np.array(particle_obstacles)))
            delta = np.abs(self.robot.pose[2] - self.particles[i].pose[2]) 
            delta = delta/np.radians(360)
            prob_obs = total_equal / len(obstacles)
            prob_head = 1-delta
            weight_heading = 0.6
            weight_landmark = 1 - weight_heading
            prob = (weight_heading*prob_head) + (weight_landmark*prob_obs) 
            correlations.append(prob)
        return correlations

    def update_particles(self, correlations):
        for i in range(self.n_particles):
            self.particles[i].probability = self.particles[i].probability * correlations[i]
        self.normalize_probability()

    def get_max_prob(self):
        max_prob = 0.0001
        max_index = -1
        for i in range(self.n_particles):
            if self.particles[i].probability > max_prob:
                max_prob = self.particles[i].probability
                max_index = i
        return max_prob, max_index

    def normalize_probability(self):
        max_prob, _ = self.get_max_prob()
        for i in range(self.n_particles):
            self.particles[i].probability = self.particles[i].probability / max_prob
            epsilon = 0.000001
            if self.particles[i].probability < epsilon:
                self.particles[i].probability = epsilon

    def resampling_particles(self, random_percents=0.2):
        list_prob = []
        for i in range(self.n_particles):
            list_prob.append(self.particles[i].probability)
        norm_prob = np.array(list_prob)/np.sum(list_prob)
        new_index = []
        for i in range(self.n_particles):
            index = np.random.choice(np.arange(0,self.n_particles), p=norm_prob)
            new_index.append(index)

        # Old particles pose and prob before resampling
        old_pose_probability = []
        for i in range(self.n_particles):
            old_pose_probability.append((self.particles[i].pose, self.particles[i].probability))

        # Resampling 
        n_percent_particles = int(random_percents*self.n_particles)
        for i in range(self.n_particles-n_percent_particles):
            pose, probability = old_pose_probability[new_index[i]]
            x = pose[0]
            y = pose[1]
            theta = pose[2]

            self.particles[i].pose = pose
            self.particles[i].pose[0] = x 
            self.particles[i].pose[1] = y 
            self.particles[i].pose[2] = theta
            self.particles[i].probability = probability
        
        for i in range(n_percent_particles):
            pose, probability = old_pose_probability[new_index[i]]
            x = pose[0]
            y = pose[1]
            theta = pose[2]
            x = random.gauss(x, 10)
            y = random.gauss(y, 10)
            theta = random.gauss(theta, np.radians(10))
            self.particles[i].pose = pose
            self.particles[i].pose[0] = x 
            self.particles[i].pose[1] = y 
            self.particles[i].pose[2] = theta
            self.particles[i].probability = 1/self.n_particles

        # check valid position
        for i in range(self.n_particles):
            acceptable = False
            while not acceptable:
                px, py = self.particles[i].get_grid_pos(self.occupancy_grid)
                if px >= 0 and px < self.map.data.shape[1] and py >= 0 and py < self.map.data.shape[0]:
                    if self.map.data[py,px] == 0:
                        acceptable = True
                    else:
                        x, y, theta = self.get_random_pose()
                        self.particles[i].pose[0] = x 
                        self.particles[i].pose[1] = y
                        self.particles[i].pose[2] = theta
                        self.particles[i].probability = 1/self.n_particles
                else:
                    x, y, theta = self.get_random_pose()
                    self.particles[i].pose[0] = x 
                    self.particles[i].pose[1] = y
                    self.particles[i].pose[2] = theta
                    self.particles[i].probability = 1/self.n_particles

    def roulette_wheel(self):
        index = int(random.random()*self.n_particles)
        # index = 0
        beta = 0.0 
        max_prob, _ = self.get_max_prob()
        new_particles_index = []
        for i in range(self.n_particles):
            beta += random.random() * 2 *max_prob
            px, py = self.particles[index].get_grid_pos(self.occupancy_grid)
            accepatable = False
            while beta > self.particles[index].probability and not accepatable:
                # print("index ", index)
                if px >= 0 and px < self.map.data.shape[1] and py >= 0 and py < self.map.data.shape[0]:
                    if self.map.data[py, px] == 0:
                        accepatable = True
                beta -= self.particles[index].probability
                index = (index+1) % self.n_particles
            new_particles_index.append(index)
            self.particles[i].pose = self.particles[index].pose
            self.particles[i].probability = self.particles[index].probability

    def render(self):
        self.screen.fill(self.field_color, pygame.Rect(0,0,self.map_width,self.map_height))
        self.screen.fill(self.occupancy_color, pygame.Rect(self.map_width,0,self.map_width*2,self.map_height))
        self.draw_map()
        self.obstacles_sprites.draw(self.screen)
        self.robot_sprites.draw(self.screen)
        self.particles_sprites.draw(self.screen)
    
    def display(self):
        self.clock.tick(120)
        pygame.display.flip()

    def run(self, iterations=100, random_move=True, plot=True):
        # iterations = 1000
        error_hist = []
        for counter in range(iterations):
            self.render()
            self.display()
            # Update motion model
            if random_move:
                forward_cmd = random.randint(0,10)
                turn_cmd = np.radians(random.randint(0,10))
                move = np.random.choice([0, 1])
                if move == 0:
                    self.robot.move_forward(forward_cmd)
                    self.particles_move_forward(forward_cmd)
                else:
                    self.robot.turn(turn_cmd)
                    self.particles_turn(turn_cmd)
            else:
                press = False
                while not press:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                self.robot.move_forward(10)
                                self.particles_move_forward(10)
                                press = True
                            elif event.key == pygame.K_DOWN:
                                self.robot.move_forward(-10)
                                self.particles_move_forward(-10)
                                press = True
                            elif event.key == pygame.K_LEFT:
                                self.robot.turn(np.radians(10))
                                self.particles_turn(np.radians(10))
                                press = True
                            elif event.key == pygame.K_RIGHT:
                                self.robot.turn(np.radians(-10))
                                self.particles_turn(np.radians(-10))
                                press = True

            # Measurement
            distances, scan_angles, obstacles = self.check_sensor(max_length=300)
            correlations = self.get_correlation(distances, scan_angles, obstacles)
            # print(correlations)
            self.update_particles(correlations)
            max_prob, index = self.get_max_prob()
            # print(max_prob, index)
            self.estimates.pose[0] = self.particles[index].pose[0]
            self.estimates.pose[1] = self.particles[index].pose[1]
            self.estimates.pose[2] = self.particles[index].pose[2]
            distance = euclidean([self.estimates.pose[0], self.estimates.pose[1]], [self.robot.pose[0], self.robot.pose[1]]) / 100
            error_hist.append(distance)
            print("Distance Error :", distance, "meters")
            self.estimates.redraw()
            # Resampling
            self.resampling_particles(random_percents=0.1)
            self.display()
            time.sleep(0.1)
        if plot:
            style.use('seaborn-poster') 
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,iterations), error_hist)
            ax.set_title("Distance Error VS Iterations")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Distance Error (m)")
            plt.show()
def main():
    env = ObstacleRunEnv(n_particles=100, n_obstacles=10)
    env.run(iterations=500, random_move=True)

if __name__ == "__main__":
    main()