import cv2
import heapq
import numpy as np

class Util:

    @staticmethod
    def get_color_based_on_activation(value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_value = max(0, min(normalized_value, 1)) 
        red = int(255 * normalized_value)
        return (red, 0, 0)

    @staticmethod
    def identify_walls(map_image):
        print('Carregando parede..')
        img = cv2.imread(map_image)

        if np.__name__ == 'cupy':
            img = np.asarray(img)
            wall_pixels = np.all(img == np.array([110, 110, 110]), axis=-1)
        else:
            wall_pixels = np.all(img == [110, 110, 110], axis=-1)



        # Inverte o eixo y do array de pixels da parede
        wall_pixels_inverted_y = np.flipud(wall_pixels)

        # Cria um array booleano representando paredes (False) e espaços livres (True)
        walls = np.where(wall_pixels_inverted_y, False, True)

        # Transpor a matriz para que possamos acessá-la usando walls[x, y]
        walls = walls.T

        print('Parede carregada !')

        return walls

    @staticmethod
    def add_sprite_to_walls( sprite, walls):

        left = int(sprite.center_x - sprite.width  / 2)
        right = int(sprite.center_x + sprite.width  / 2)
        bottom = int(sprite.center_y - sprite.height  / 2)
        top = int(sprite.center_y + sprite.height  / 2)

        # Cria uma máscara para essa área
        mask = np.zeros_like(walls, dtype=bool)
        mask[left:right, bottom:top] = True

        # Aplica a máscara em 'walls', marcando a área como parede (True)
        walls = np.where(mask, True, walls)

        return walls


    def dijkstra(map_walls, start, goal):

        rows, cols = map_walls.shape
        distances = np.full((rows, cols), np.inf)
        distances[start] = 0

        queue = [(0, start)]

        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        visited = np.zeros_like(map_walls, dtype=bool)

        while queue:
            current_distance, (x, y) = heapq.heappop(queue)
            visited[x, y] = True

            if (x, y) == goal:
                break

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and map_walls[nx, ny] == 0:
                    distance = current_distance + 1
                    if distance < distances[nx, ny]:
                        distances[nx, ny] = distance
                        heapq.heappush(queue, (distance, (nx, ny)))

        distances = np.where(distances == np.inf, np.nan, distances)

        return distances