//This code is inspired by maxontech
import sys
import pygame
import neat
import math
import os

pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 1244, 1016
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
TRACK = pygame.image.load("img/track.png")

class AutonomousCar(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("img/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.velocity_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_velocity = 5
        self.turn_direction = 0
        self.alive = True
        self.sensors = []

    def update(self):
        self.sensors.clear()
        self.drive()
        self.rotate()
        for sensor_angle in (-60, -30, 0, 30, 60):
            self.sensor(sensor_angle)
        self.handle_collision()
        self.process_data()

    def drive(self):
        self.rect.center += self.velocity_vector * 6

    def handle_collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.turn_direction == 1:
            self.angle -= self.rotation_velocity
            self.velocity_vector.rotate_ip(self.rotation_velocity)
        if self.turn_direction == -1:
            self.angle += self.rotation_velocity
            self.velocity_vector.rotate_ip(-self.rotation_velocity)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def sensor(self, sensor_angle):
        distance_to_obstacle = 0
        current_x = int(self.rect.center[0])
        current_y = int(self.rect.center[1])

        while distance_to_obstacle < 200 and not SCREEN.get_at((current_x, current_y)) == pygame.Color(2, 105, 31, 255):
            distance_to_obstacle += 1
            current_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + sensor_angle)) * distance_to_obstacle)
            current_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + sensor_angle)) * distance_to_obstacle)

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (current_x, current_y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (current_x, current_y), 3)

        distance_from_center = int(math.sqrt(math.pow(self.rect.center[0] - current_x, 2)
                                            + math.pow(self.rect.center[1] - current_y, 2)))

        self.sensors.append([sensor_angle, distance_from_center])

    def process_data(self):
        input_data = [0, 0, 0, 0, 0]
        for i, sensor in enumerate(self.sensors):
            input_data[i] = int(sensor[1])
        return input_data

def remove_car(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)

def evaluate_genomes(genomes, config):
    global cars, ge, nets
    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(AutonomousCar()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run_simulation = True
    while run_simulation:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove_car(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.process_data())
            if output[0] > 0.7:
                car.sprite.turn_direction = 1
            if output[1] > 0.7:
                car.sprite.turn_direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.turn_direction = 0

        SCREEN.blit(TRACK, (0, 0))

        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()

def run_neat(config_path):
    global population
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.run(evaluate_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run_neat(config_path)
