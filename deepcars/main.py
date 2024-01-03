import copy
import arcade
import math
import numpy as np
import pickle
import numpy
from utils import Util
from queue import Queue
from threading import Thread


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
MAP_WIDTH = 2000
MAP_HEIGHT = 2000
TOTAL_CARS = 200
TRAINING_BLANK = True
MIN_CAR_SPEED = 1
MAX_CAR_SPEED = 10
CAR_ROTATION_SPEED = 5
MAP_INITIAL_POS = (1875 , 1200)
MAP_FINAL_POS =(1346,266)
MAP_MAX_DISTANCE = 14100
INITIAL_DESTROYER_SPEED = MIN_CAR_SPEED
BRAIN_SIZE_INPUT = 18
BRAIN_SIZE_HIDDEN = 6
BRAIN_SIZE_OUTPUT = 4
RAY_CAR_COUNT = BRAIN_SIZE_INPUT
SPIKES_POS = [
        [310, 1868],
        [705, 1868],
        [500, 1930],
        [900, 1930],
        [215, 467],
        [240, 410],
        [280, 369],
        [337, 350],
        [335, 244],
        [392, 245],
        [434, 200],
        [427, 150],
        [1385, 360],
        [1315, 470],
        [1385, 580],
        [1315, 690],
        [1385, 800],
        [1315, 910],
        [1385, 1020],
        [1315, 1130],
        [1865, 825],
        [1560, 530],
        [1930, 235]
    ]
SPEED_ZONES_POS = [
    [1520, 1610, False, 0],
    [130, 1230, True, 270],
    [790, 350,  False, 0],
    [790, 900,  False, 0],
    [669, 630, True, 90],
    [1042, 965, True, 270],
    [1164, 600, False, 90],
    [1040, 265, False, 90],
]

arcade.enable_timings()

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except:
                pass
            finally:
                self.tasks.task_done()


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


class GameData(object):
    def __init__(self):
        self.brain = None
        self.generation = None
        self.processing_seconds = 0

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)




class NeuralNetwork:

    def __init__(self, layer_sizes, activation='leaky_relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.activation = activation
        self.layer_outputs = []

    def mutate_randomly(self):
        for layer in range(len(self.weights)):
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    self.weights[layer][i, j] += np.random.randn()

        for layer in range(len(self.biases)):
            for i in range(self.biases[layer].shape[0]):
                self.biases[layer][i, 0] += np.random.randn()

    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x,)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, x * 0.01)   
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        self.layer_outputs = []

        for b, w in zip(self.biases, self.weights):
            x = self.activate(np.dot(w, x) + b)
            self.layer_outputs.append(x.copy())

        return x

    def clone(self):
        return copy.deepcopy(self)


class GameContext:
    def __init__(self):
        self.walls = None

class Spike(arcade.Sprite):
    def __init__(self, image_path, scale=1, x=0, y=0, angle=0):
        super().__init__(image_path, scale)
        self.center_x = x
        self.center_y = y
        self.angle = angle

class SpeedZone(arcade.Sprite):
    def __init__(self, image_path, scale=1, x=0, y=0, angle=0, increase=True):
        super().__init__(image_path, scale)
        self.center_x = x
        self.center_y = y
        self.angle = angle
        self.increase = increase

class Car(arcade.Sprite):
    def __init__(self, game_context, image_path, id=-1, scale=1, x=0, y=0, angle=0):
        super().__init__(image_path, scale)
        self.id = id
        self.game_context = game_context
        self.center_x = x
        self.center_y = y
        self.angle = angle
        self.rotating_left = False
        self.rotating_right = False
        self.moving = False
        self.raycasting = RayCasting(self,RAY_CAR_COUNT,180)
        self.collided = False
        self.distance = 0
        self.brain = NeuralNetwork([BRAIN_SIZE_INPUT, BRAIN_SIZE_HIDDEN, BRAIN_SIZE_HIDDEN, BRAIN_SIZE_OUTPUT],'relu')
        self.manual = False
        self.speed = MIN_CAR_SPEED

    def rotate_left(self):
        self.angle += CAR_ROTATION_SPEED
        self.angle = self.angle % 360

        if self.moving:
            self.forward()

    def rotate_right(self):
        self.angle -= CAR_ROTATION_SPEED
        self.angle = self.angle % 360

        if self.moving:
            self.forward()

    def speed_up(self):
        if self.speed < MAX_CAR_SPEED:
            self.speed += 0.5

    def slow_down(self):
        if self.speed > MIN_CAR_SPEED:
            self.speed -= 0.5

    def forward(self):
        self.moving = True
        angle_rad = math.radians(self.angle)
        self.change_x = self.speed * math.cos(angle_rad)
        self.change_y = self.speed * math.sin(angle_rad)

    def stop(self):
        self.moving = False
        self.change_x = 0
        self.change_y = 0

    def update(self):

        if not self.collided:

            if self.manual:
                if self.rotating_right:
                    self.rotate_right()

                if self.rotating_left:
                    self.rotate_left()

            else:
                sensor_input = np.array(self.raycasting.ray_distance).reshape(-1, 1)

                decision = self.brain.forward(sensor_input)

                if decision[0] > 0:
                    self.speed_up()

                if decision[1] > 0:
                    self.slow_down()

                if decision[2] > 0:
                    self.rotate_right()

                if decision[3] > 0:
                    self.rotate_left()

            self.forward()

            cur_dist = self.game_context.map_distances[ int(self.center_x), int(self.center_y) ]

            if not math.isnan(cur_dist):
                self.distance = cur_dist
            
            self.raycasting.cast_rays(self.game_context.walls)

            if self.game_context.speed_zones[int(self.center_x), int(self.center_y)] == -1:
                self.speed = MIN_CAR_SPEED
            elif self.game_context.speed_zones[int(self.center_x), int(self.center_y)] == 1:
                self.speed = MAX_CAR_SPEED


            if(self.raycasting.min_distance <= 1):
                self.collided = True

        return super().update()

    def draw(self, *, filter=None, pixelated=None, blend_function=None):
        r = super().draw(filter=filter, pixelated=pixelated, blend_function=blend_function)
        
        if self.game_context.model_success:
            self.raycasting.draw()

        return r

class RayCasting:
    def __init__(self, car, num_rays=18, max_distance=200):
        self.num_rays = num_rays
        self.ray_distance = [max_distance] * num_rays
        self.ray_start_points = [(0, 0)] * num_rays
        self.ray_end_points = [(0, 0)] * num_rays
        self.car = car
        self.max_distance = max_distance
        self.min_distance = max_distance

    def cast_rays(self, walls: numpy.ndarray):
        for i in range(self.num_rays):
            angle = self.car.angle - 90.0 + (i)*360.0/((RAY_CAR_COUNT-2))
            self.cast_single_ray(i, angle, walls)
            self.min_distance = min(self.min_distance, self.ray_distance[i])

    def encontrar_intersecao(raio_origem, raio_direcao, segmento_inicio, segmento_fim):
        """
        Calcula o ponto de interseção entre um raio e um segmento de linha.
        Retorna None se não houver interseção.
        """
        # Vetor direção do segmento
        segmento_direcao = segmento_fim - segmento_inicio

        # Matriz do sistema linear
        matriz = np.array([-raio_direcao, segmento_direcao]).T

        # Verifica se os vetores são paralelos (determinante = 0)
        if np.linalg.det(matriz) == 0:
            return None

        # Encontra os coeficientes t e u
        t, u = np.linalg.solve(matriz, segmento_inicio - raio_origem)

        # Verifica se a interseção está nos segmentos
        if 0 <= t and 0 <= u <= 1:
            return raio_origem + t * raio_direcao
        return None


    def cast_single_ray(self, idx, angle, walls):

        angle_rad = math.radians(angle)
        start_x, start_y = self.car.center_x, self.car.center_y

        if walls[int(start_x), int(start_y)]:
            self.set_ray_data(idx, start_x, start_y, start_x,start_y,0)
            return

        step_size = 10
        reached = False

        for distance in range(0, self.max_distance, step_size):
            end_x = start_x + distance * math.cos(angle_rad)
            end_y = start_y + distance * math.sin(angle_rad)

            if walls[int(end_x), int(end_y)]:
                reached = True
                self.set_ray_data(idx, start_x, start_y, end_x, end_y, distance)
                break

        if not reached:
            self.set_ray_data(idx, start_x, start_y, end_x, end_y, distance)


    def set_ray_data(self, idx, start_x, start_y, end_x, end_y, distance):
        self.ray_start_points[idx] = (start_x, start_y)
        self.ray_end_points[idx] = (end_x, end_y)
        self.ray_distance[idx] = distance

    def draw(self):
        for i in range(self.num_rays):            
            arcade.draw_line(self.ray_start_points[i][0], self.ray_start_points[i][1], self.ray_end_points[i][0], self.ray_end_points[i][1], arcade.color.RED, 1)
            arcade.draw_circle_filled(self.ray_end_points[i][0], self.ray_end_points[i][1],6,arcade.color.RED)


class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Arcade Car Game")
        self.physics_engine = None
        self.game_context = GameContext()
        self.map_width = SCREEN_WIDTH
        self.map_height = SCREEN_HEIGHT
        self.zoom_level = 2
        self.camera_x = SCREEN_WIDTH / 2 +1000
        self.camera_y = SCREEN_HEIGHT / 2+1000
        self.best_brain = None
        self.game_context.map_distances = None
        self.ui_camera = arcade.Camera(self.width, self.height)
        self.game_camera = arcade.Camera(self.width, self.height)
        self.text_messages = [None] * 30
        self.cars = None
        self.best_car = None
        self.destroyer_distance = 0
        self.destroyer_distance_points = []
        self.destroyer_speed = INITIAL_DESTROYER_SPEED
        self.generation = 1
        self.best_distance_ever = 0
        self.game_context.model_success = False
        self.spikes = []
        self.speed_zone_sprites = []
        self.game_context.speed_zones = np.full((MAP_WIDTH, MAP_HEIGHT), 0)
        self.game_data = GameData()
        self.processing_seconds = 0
        self.pool = ThreadPool(50)


    def setup(self):
        self.map_sprite = arcade.Sprite("imagens/pistaObstaculo.png",1)
        self.map_sprite.center_x = MAP_WIDTH // 2
        self.map_sprite.center_y = MAP_HEIGHT // 2

        self.game_context.walls = Util.identify_walls("imagens/pistaObstaculo.png")

        for pos in SPIKES_POS:
            spike = Spike("imagens/spike.png",0.5, pos[0], pos[1], 0)
            self.game_context.walls = Util.add_sprite_to_walls(spike, self.game_context.walls)
            self.spikes.append(spike)

        for pos in SPEED_ZONES_POS:
            inc = pos[2]
            img = "imagens/turbo.png" if inc else "imagens/lama.png"
            sz = SpeedZone(img,0.5, pos[0], pos[1],pos[3],inc)
            self.speed_zone_sprites.append(sz)
            x = int(sz.center_x - (sz.width/2))
            y = int(sz.center_y - (sz.height/2))
            dx = int(sz.center_x + (sz.width/2))
            dy = int(sz.center_y + (sz.height/2))
            self.game_context.speed_zones[ x:dx, y:dy ] = 1 if inc else -1 

        try:        
            self.game_context.map_distances = np.load('distance_matrix.npy')
        except:
            pass

        if self.game_context.map_distances is None:
            self.game_context.map_distances = Util.dijkstra(self.game_context.walls, MAP_INITIAL_POS, MAP_FINAL_POS)
            np.save('distance_matrix.npy', self.game_context.map_distances)


        try:        
            gd = GameData.load('game_data.dc')
            self.best_brain = gd.brain
            self.processing_seconds = gd.processing_seconds
            self.generation = gd.generation
        except:
            pass


        self.post_setup()

    def post_setup(self):
        self.cars = []
        self.destroyer_distance = 0
        self.destroyer_speed = INITIAL_DESTROYER_SPEED

        if self.game_context.model_success:
            i=0
            car = Car(self.game_context,'imagens/carro' + str(i % 6) + '.png',i, 0.05, MAP_INITIAL_POS[0] , MAP_INITIAL_POS[1] , 90)
            car.brain = self.best_brain.clone()
            self.best_brain = None
            self.cars.append(car)
            self.set_best_car(car)
        else:
            
            for i in range(TOTAL_CARS):
                car = Car(self.game_context,'imagens/carro' + str(i % 6) + '.png',i, 0.05, MAP_INITIAL_POS[0] , MAP_INITIAL_POS[1] , 90)
                
                if(self.best_brain):
                    car.brain = self.best_brain.clone()
                    car.brain.mutate_randomly()

                self.cars.append(car)

            self.best_brain = None
            self.set_best_car(None)

    def update_viewport(self):

            if self.best_car:
                self.camera_x = self.best_car.center_x
                self.camera_y = self.best_car.center_y
                self.game_camera.move_to( (self.best_car.center_x - (self.width / 2), self.best_car.center_y - (self.height / 2)) , 0.2)


    def on_update(self, delta_time):

        if not self.game_context.model_success:
            self.processing_seconds += delta_time

        self.text_messages = [None] * 30

        self.update_cars()
    
        alive_cars = sum(1 for car in self.cars if not car.collided)

        self.update_destroyer()

        for spike in self.spikes:
            spike.angle += 3
            spike.angle = spike.angle % 360

        if self.best_car:
            self.text_messages[0]= f"Melhor carro #{self.best_car.id:02d} - Vel: #{self.best_car.speed:03.02f} - Dist: {int(self.best_car.distance)} ({self.percent_distance(self.best_car.distance):03.02f}%) "

        self.text_messages[1] = f'Carros vivos: '+str(alive_cars)

        self.text_messages[2] = f'Geração: {self.generation}'

        self.text_messages[3] = f'Destruidor: Vel - #{self.destroyer_speed:03.02f} - Dist - {self.destroyer_distance:05.02f} ({self.percent_distance(self.destroyer_distance):03.02f}%)'

        self.text_messages[4] = f'Melhor distancia: {self.best_distance_ever:05.02f} ({self.percent_distance(self.best_distance_ever):03.02f}%)'

        hh = int(self.processing_seconds) // 3600
        mm = (int(self.processing_seconds) % 3600) // 60
        ss = int(self.processing_seconds) % 60

        self.text_messages[6] = f'Tempo: {hh:02d}:{mm:02d}:{ss:02d}'

        if alive_cars == 0:
            self.end_generation()
            self.post_setup()

        self.update_viewport()

    def update_destroyer(self):
        self.destroyer_distance = self.destroyer_distance + self.destroyer_speed
        if self.destroyer_speed <= MAX_CAR_SPEED*0.8:
            self.destroyer_speed = self.destroyer_speed + 0.008

        if self.game_context.model_success:
            self.destroyer_distance_points = self.find_laser_position()

    def on_draw(self):

        arcade.start_render()
        
        if (TRAINING_BLANK and self.game_context.model_success) or not TRAINING_BLANK:
            self.game_camera.use()
            
            self.map_sprite.draw()
            
            for sz in self.speed_zone_sprites:
                sz.draw()

            for car in self.cars:
                car.draw()
                if car == self.best_car:
                    arcade.draw_circle_filled(car.center_x,car.center_y,3,arcade.color.WHITE)
            
            for obs in self.spikes:
                obs.draw()

            if self.game_context.model_success:
                arcade.draw_line_strip(self.destroyer_distance_points, arcade.color.GREEN,3)

        self.ui_camera.use()


        if arcade.timings_enabled():
            fps = arcade.get_fps()
            fps_text = "FPS: %.2f" % (fps)
            self.text_messages[5] = fps_text

        for idx, m in enumerate(self.text_messages):
            if m is not None:
                arcade.draw_text(m, 10, self.height - (20 * (idx+1)), arcade.color.WHITE, 14)

        if self.best_car and self.game_context.model_success:
                self.draw_neural_network(self.best_car.brain, position=(SCREEN_WIDTH-390, SCREEN_HEIGHT-10))

        arcade.finish_render()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom_level += scroll_y * 0.1  # Ajuste este valor conforme necessário
        self.zoom_level = max(0.1, min(self.zoom_level, 5))  # Limitar o zoom entre 10% e 500%
        self.update_viewport()
            
    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.ui_camera.resize(width, height)


    def on_key_press(self, key, modifiers):
        self.best_car.manual = True
        if key == arcade.key.LEFT:
            self.best_car.rotating_left = True
            self.best_car.rotating_right = False
        if key == arcade.key.RIGHT:
            self.best_car.rotating_left = False
            self.best_car.rotating_right = True
        if key == arcade.key.UP:
            self.best_car.forward()
        if key == arcade.key.SPACE:
            global TRAINING_BLANK
            TRAINING_BLANK = not TRAINING_BLANK

    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP:
            self.best_car.stop()
        if key == arcade.key.LEFT:
            self.best_car.rotating_left = False
        if key == arcade.key.RIGHT:
            self.best_car.rotating_right = False
        pass



    def draw_neural_network(self, neural_network, position, layer_width=100, neuron_radius=6):
        x, y = position
        spacing = 15

        for layer_idx in range(1, len(neural_network.layer_sizes)):
            current_x = x + layer_idx * layer_width
            prev_layer_size = neural_network.layer_sizes[layer_idx - 1]
            for neuron_idx in range(neural_network.layer_sizes[layer_idx]):
                current_y = y - neuron_idx * spacing
                for prev_neuron_idx in range(prev_layer_size):
                    prev_x = x + (layer_idx - 1) * layer_width
                    prev_y = y - prev_neuron_idx * spacing

                    activation = neural_network.layer_outputs[layer_idx - 1][neuron_idx, 0]
                    line_color = arcade.color.RED if activation > 0 else arcade.color.WHITE

                    arcade.draw_line(prev_x, prev_y, current_x, current_y, line_color, 1)

        for layer_idx, layer_size in enumerate(neural_network.layer_sizes):
            current_x = x + layer_idx * layer_width
            for neuron_idx in range(layer_size):
                current_y = y - neuron_idx * spacing

                if layer_idx == 0:
                    sensor_value = self.best_car.raycasting.ray_distance[neuron_idx]
                    neuron_color = Util.get_color_based_on_activation(sensor_value, 0, 200)
                elif layer_idx > 0:
                    weight = neural_network.layer_outputs[layer_idx - 1][neuron_idx,0]
                    neuron_color = arcade.color.RED if weight > 0 else arcade.color.BLACK                    
                
                if layer_idx == layer_size - 1:
                    text = ""
                    if neuron_idx == 0:
                        text = "Acelerar"
                    elif neuron_idx == 1:
                        text = "Frear"
                    elif neuron_idx == 2:
                        text = "Direita"
                    elif neuron_idx == 3:
                        text = "Esquerda"

                    arcade.draw_text(text, current_x+10, current_y-5,arcade.color.WHITE)


                arcade.draw_circle_filled(current_x, current_y, neuron_radius, neuron_color)
                arcade.draw_circle_outline(current_x, current_y, neuron_radius, arcade.color.WHITE, 1)
    
    def set_best_car(self, best_car):
        self.best_car = best_car

    def percent_distance(self, cur_distance):
        return (cur_distance / MAP_MAX_DISTANCE) * 100

    def end_generation(self):

        best_car = max(self.cars, key=lambda car: car.distance)

        self.best_distance_ever = max(self.best_distance_ever, best_car.distance)

        best_brain = best_car.brain.clone()

        self.best_brain = best_brain

        if self.best_distance_ever >= MAP_MAX_DISTANCE:
            self.game_context.model_success = True
        else:
            self.generation = self.generation + 1

        self.game_data.brain = self.best_brain
        self.game_data.generation = self.generation
        self.game_data.processing_seconds = self.processing_seconds

        self.game_data.save('game_data.dc')


    def find_laser_position(self):
        condition = self.game_context.map_distances == int(self.destroyer_distance)
        return np.argwhere(condition)

    def update_car(self, car):
        if not car.collided:

            car.update()

            if self.destroyer_distance > car.distance:
                car.collided = True

            if self.best_car is None or (car.distance > self.best_car.distance and not car.collided):
                self.set_best_car(car)


    def update_cars(self):

        for car in self.cars:
            self.pool.add_task(self.update_car, car)

        self.pool.wait_completion()


def main():
    game = MyGame()
    game.setup()
    arcade.run()

if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats = pstats.Stats(profiler)
    stats.dump_stats('deepcars.prof')
