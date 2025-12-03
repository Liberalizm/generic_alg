import pygame
import Box2D
from Box2D import b2World
import math
import random
import collections
from typing import List, Dict, Optional
import multiprocessing as mp
from multiprocessing import Process, Array, Value
import ctypes
import time
#
# from nn import SimpleNeuralNetwork
from test_sim import LegExtensionTestController as SimpleNeuralNetwork
from cat import Cat

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
PPM = 20.0  # pixels per meter
TARGET_FPS = 60

# Colors
SKY_COLOR = (135, 206, 235)
GROUND_COLOR = (80, 160, 80)
GRASS_COLOR = (60, 140, 60)
RULER_COLOR = (50, 50, 50)
CAT_BODY_COLOR = (100, 100, 110)
CAT_LEG_COLOR = (90, 90, 100)
MUSCLE_COLOR = (200, 80, 80)

# UI Colors
PANEL_BG_COLOR = (70, 85, 95)
SLIDER_BG_COLOR = (60, 70, 80)
SLIDER_FILL_COLOR = (100, 180, 100)
SLIDER_LOW_COLOR = (200, 100, 100)
SLIDER_MEDIUM_COLOR = (200, 180, 100)
SLIDER_BORDER_COLOR = (40, 50, 60)
GRAPH_LINE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
    (255, 100, 255), (100, 255, 255), (255, 180, 100), (180, 100, 255),
]

MUSCLE_NAMES = [
    "FL Hip", "FL Knee", "FR Hip", "FR Knee",
    "BL Hip", "BL Knee", "BR Hip", "BR Knee",
]

# Parallel simulation settings
NUM_CORES = 20          # Number of CPU cores to use
CATS_PER_CORE = 2      # Number of cats simulated on each core
NUM_CATS = NUM_CORES * CATS_PER_CORE

ITERATION_TIME = 15.0

# Shared memory layout per cat (floats):
# 0: body_x, 1: body_y, 2: body_angle
# 3-18: body_vertices (4 vertices * 2 coords = 8) + padding
# 19-82: leg_vertices (4 legs * 2 parts * 4 vertices * 2 coords = 64)
# 83-114: muscle_data (8 muscles * 4 values: ax, ay, bx, by) = 32
# 115-122: muscle_activations (8)
# 123-130: muscle_energies (8)
# 131: distance, 132: speed
CAT_DATA_SIZE = 140


def simulation_worker(core_id: int, num_cats: int, shared_data: Array,
                      running_flag: Value, reset_flag: Value,
                      spawn_x: float, ground_y: float):
    """Worker process running continuous simulation."""

    world = b2World(gravity=(0, -10), doSleep=True)

    ground_body = world.CreateStaticBody(position=(0, ground_y - 0.5))
    ground_body.CreatePolygonFixture(
        box=(1000, 0.5),
        categoryBits=0x0001,
        maskBits=0xFFFF
    )

    cats: List[Cat] = []

    def create_cats():
        nonlocal cats
        for cat in cats:
            world.DestroyBody(cat.body)
            for leg in cat.legs:
                world.DestroyBody(leg['upper'])
                world.DestroyBody(leg['lower'])
        cats.clear()

        for i in range(num_cats):
            # Use local index for collision filtering within this worker's world
            cat = Cat(world, position=(spawn_x, ground_y + 15), cat_index=i)
            nn = SimpleNeuralNetwork(num_muscles=len(cat.muscles))
            cat.set_neural_network(nn)
            cats.append(cat)

    create_cats()

    dt = 1.0 / TARGET_FPS
    last_time = time.time()

    while running_flag.value:
        # Check reset flag
        if reset_flag.value:
            create_cats()
            reset_flag.value = 0

        # Timing control
        current_time = time.time()
        elapsed = current_time - last_time

        if elapsed >= dt:
            last_time = current_time

            # Step physics
            world.Step(dt, 8, 3)

            # Update cats and write to shared memory
            for i, cat in enumerate(cats):
                cat.update(dt)

                global_idx = core_id * num_cats + i
                offset = global_idx * CAT_DATA_SIZE

                # Body position and angle
                shared_data[offset + 0] = cat.body.position.x
                shared_data[offset + 1] = cat.body.position.y
                shared_data[offset + 2] = cat.body.angle

                # Body vertices
                idx = offset + 3
                for fixture in cat.body.fixtures:
                    for v in fixture.shape.vertices:
                        world_v = cat.body.transform * v
                        shared_data[idx] = world_v.x
                        shared_data[idx + 1] = world_v.y
                        idx += 2

                # Leg vertices
                idx = offset + 19
                for leg in cat.legs:
                    for part in ['upper', 'lower']:
                        body = leg[part]
                        for fixture in body.fixtures:
                            for v in fixture.shape.vertices:
                                world_v = body.transform * v
                                shared_data[idx] = world_v.x
                                shared_data[idx + 1] = world_v.y
                                idx += 2

                # Muscle anchors
                idx = offset + 83
                for muscle in cat.muscles:
                    anchor_a, anchor_b = muscle.get_world_anchors()
                    shared_data[idx] = anchor_a.x
                    shared_data[idx + 1] = anchor_a.y
                    shared_data[idx + 2] = anchor_b.x
                    shared_data[idx + 3] = anchor_b.y
                    idx += 4

                # Muscle activations
                idx = offset + 115
                for muscle in cat.muscles:
                    shared_data[idx] = muscle.activation
                    idx += 1

                # Muscle energies
                idx = offset + 123
                for muscle in cat.muscles:
                    shared_data[idx] = muscle.get_energy_percent()
                    idx += 1

                # Stats
                shared_data[offset + 131] = cat.stats.get_distance_from_spawn(cat.get_position().x)
                shared_data[offset + 132] = cat.stats.get_current_speed()
        else:
            time.sleep(0.0001)


def read_cat_data(shared_data: Array, cat_idx: int) -> dict:
    """Read cat data from shared memory."""
    offset = cat_idx * CAT_DATA_SIZE

    data = {
        'body_position': (shared_data[offset], shared_data[offset + 1]),
        'body_angle': shared_data[offset + 2],
        'body_vertices': [],
        'leg_data': [],
        'muscle_data': [],
        'muscle_energies': [],
        'distance': shared_data[offset + 131],
        'speed': shared_data[offset + 132],
    }

    # Body vertices (4 vertices)
    idx = offset + 3
    vertices = []
    for _ in range(4):
        vertices.append((shared_data[idx], shared_data[idx + 1]))
        idx += 2
    data['body_vertices'] = [vertices]

    # Leg vertices (4 legs, 2 parts each, 4 vertices each)
    idx = offset + 19
    for _ in range(4):
        leg_info = {'upper_vertices': [], 'lower_vertices': []}
        for part in ['upper_vertices', 'lower_vertices']:
            vertices = []
            for _ in range(4):
                vertices.append((shared_data[idx], shared_data[idx + 1]))
                idx += 2
            leg_info[part] = [vertices]
        data['leg_data'].append(leg_info)

    # Muscle data
    idx = offset + 83
    act_idx = offset + 115
    for _ in range(8):
        data['muscle_data'].append({
            'anchor_a': (shared_data[idx], shared_data[idx + 1]),
            'anchor_b': (shared_data[idx + 2], shared_data[idx + 3]),
            'activation': shared_data[act_idx],
        })
        idx += 4
        act_idx += 1

    # Muscle energies
    idx = offset + 123
    for _ in range(8):
        data['muscle_energies'].append(shared_data[idx])
        idx += 1

    return data


def world_to_screen(x, y, camera_x):
    screen_x = (x - camera_x) * PPM + SCREEN_WIDTH / 2
    screen_y = SCREEN_HEIGHT - y * PPM
    return int(screen_x), int(screen_y)


def draw_cat_from_render_data(surface, render_data: dict, camera_x):
    for vertices in render_data['body_vertices']:
        screen_vertices = [world_to_screen(v[0], v[1], camera_x) for v in vertices]
        pygame.draw.polygon(surface, CAT_BODY_COLOR, screen_vertices)
        pygame.draw.polygon(surface, (50, 50, 50), screen_vertices, 2)

    for leg_info in render_data['leg_data']:
        for vertices in leg_info['upper_vertices']:
            screen_vertices = [world_to_screen(v[0], v[1], camera_x) for v in vertices]
            pygame.draw.polygon(surface, CAT_LEG_COLOR, screen_vertices)
            pygame.draw.polygon(surface, (50, 50, 50), screen_vertices, 2)
        for vertices in leg_info['lower_vertices']:
            screen_vertices = [world_to_screen(v[0], v[1], camera_x) for v in vertices]
            pygame.draw.polygon(surface, CAT_LEG_COLOR, screen_vertices)
            pygame.draw.polygon(surface, (50, 50, 50), screen_vertices, 2)

    for muscle_info in render_data['muscle_data']:
        pos_a = world_to_screen(muscle_info['anchor_a'][0], muscle_info['anchor_a'][1], camera_x)
        pos_b = world_to_screen(muscle_info['anchor_b'][0], muscle_info['anchor_b'][1], camera_x)
        intensity = int(100 + 155 * muscle_info['activation'])
        thickness = max(2, int(4 * muscle_info['activation']))
        pygame.draw.line(surface, (5, 50, 50), pos_a, pos_b, thickness)


def draw_ruler(surface, camera_x, ground_y):
    screen_ground_y = SCREEN_HEIGHT - ground_y * PPM
    left_world = camera_x - SCREEN_WIDTH / (2 * PPM)
    right_world = camera_x + SCREEN_WIDTH / (2 * PPM)
    start_mark = int(left_world) - 1
    end_mark = int(right_world) + 2
    font = pygame.font.SysFont('Arial', 14)

    for meter in range(start_mark, end_mark):
        screen_x = int((meter - camera_x) * PPM + SCREEN_WIDTH / 2)
        if 0 <= screen_x <= SCREEN_WIDTH:
            if meter % 5 == 0:
                pygame.draw.line(surface, RULER_COLOR, (screen_x, screen_ground_y - 15), (screen_x, screen_ground_y + 5), 2)
                text = font.render(f"{meter}m", True, RULER_COLOR)
                surface.blit(text, (screen_x - 15, screen_ground_y - 30))
            else:
                pygame.draw.line(surface, RULER_COLOR, (screen_x, screen_ground_y - 8), (screen_x, screen_ground_y + 3), 1)


def draw_grass(surface, camera_x, ground_y):
    screen_ground_y = SCREEN_HEIGHT - ground_y * PPM
    random.seed(42)
    for i in range(-50, SCREEN_WIDTH + 50, 8):
        world_x = camera_x + (i - SCREEN_WIDTH / 2) / PPM
        height = 10 + random.randint(0, 15)
        sway = math.sin(world_x * 0.5) * 3
        grass_color = (40 + random.randint(0, 30), 120 + random.randint(0, 40), 40 + random.randint(0, 30))
        pygame.draw.line(surface, grass_color, (i, screen_ground_y), (i + sway, screen_ground_y - height), 2)


def draw_flag(surface, x, camera_x, ground_y):
    screen_x, screen_y = world_to_screen(x, ground_y, camera_x)
    pole_height = 80
    pygame.draw.line(surface, (139, 90, 43), (screen_x, screen_y), (screen_x, screen_y - pole_height), 4)
    for row in range(3):
        for col in range(4):
            color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
            rect = pygame.Rect(screen_x + col * 10, screen_y - pole_height + row * 10, 10, 10)
            pygame.draw.rect(surface, color, rect)


def draw_muscle_energy_sliders(surface, muscle_energies: list, x: int, y: int):
    font = pygame.font.SysFont('Arial', 11)
    panel_height = len(muscle_energies) * 16 + 35
    panel_rect = pygame.Rect(x - 10, y - 25, 240, panel_height)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, panel_rect, 2, border_radius=8)

    title = font.render("Muscle Energy", True, (255, 255, 255))
    surface.blit(title, (x, y - 20))

    for i, energy in enumerate(muscle_energies):
        slider_y = y + i * 16
        fill_color = SLIDER_FILL_COLOR if energy > 0.5 else (SLIDER_MEDIUM_COLOR if energy > 0.25 else SLIDER_LOW_COLOR)

        bg_rect = pygame.Rect(x + 50, slider_y, 150, 12)
        pygame.draw.rect(surface, SLIDER_BG_COLOR, bg_rect, border_radius=3)
        if energy > 0:
            fill_rect = pygame.Rect(x + 50, slider_y, int(150 * energy), 12)
            pygame.draw.rect(surface, fill_color, fill_rect, border_radius=3)
        pygame.draw.rect(surface, SLIDER_BORDER_COLOR, bg_rect, 1, border_radius=3)

        name = MUSCLE_NAMES[i] if i < len(MUSCLE_NAMES) else f"M{i}"
        surface.blit(font.render(name, True, (255, 255, 255)), (x, slider_y))
        surface.blit(font.render(f"{int(energy * 100)}%", True, (255, 255, 255)), (x + 205, slider_y))


def draw_force_graph(surface, force_histories: List[collections.deque], x: int, y: int, max_samples: int):
    font = pygame.font.SysFont('Arial', 12)
    small_font = pygame.font.SysFont('Arial', 9)
    width, height = 250, 120

    panel_rect = pygame.Rect(x - 10, y - 25, width + 20, height + 55)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, panel_rect, 2, border_radius=8)

    surface.blit(font.render("Muscle Forces (5s)", True, (255, 255, 255)), (x, y - 20))

    graph_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, (50, 60, 70), graph_rect)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, graph_rect, 1)

    for i in range(5):
        pygame.draw.line(surface, (70, 80, 90), (x, y + height * i // 4), (x + width, y + height * i // 4), 1)

    max_force = 1.0
    for h in force_histories:
        if h:
            max_force = max(max_force, max(h))

    for idx, history in enumerate(force_histories):
        if len(history) < 2:
            continue
        color = GRAPH_LINE_COLORS[idx % len(GRAPH_LINE_COLORS)]
        points = [(x + width * i // max(1, max_samples - 1),
                   max(y, min(y + height, y + height - int(height * f / max_force))))
                  for i, f in enumerate(history)]
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)

    legend_y, legend_x = y + height + 5, x
    for i in range(min(len(force_histories), len(MUSCLE_NAMES))):
        pygame.draw.rect(surface, GRAPH_LINE_COLORS[i % len(GRAPH_LINE_COLORS)], (legend_x, legend_y, 8, 8))
        surface.blit(small_font.render(MUSCLE_NAMES[i], True, (200, 200, 200)), (legend_x + 10, legend_y - 2))
        legend_x += 55
        if (i + 1) % 4 == 0:
            legend_y += 12
            legend_x = x


def draw_iteration_info(surface, iteration: int, time_elapsed: float, max_time: float):
    font = pygame.font.SysFont('Arial', 24, bold=True)
    small_font = pygame.font.SysFont('Arial', 14)
    panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, 10, 100, 60)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, panel_rect, 2, border_radius=8)
    surface.blit(small_font.render("Iteration", True, (180, 180, 180)), (SCREEN_WIDTH // 2 - 25, 15))
    surface.blit(font.render(f"{iteration}", True, (255, 255, 255)), (SCREEN_WIDTH // 2 - 15, 32))
    surface.blit(small_font.render(f"{time_elapsed:.1f} / {max_time:.0f}s", True, (180, 180, 180)), (SCREEN_WIDTH // 2 - 30, 52))


def draw_best_cat_info(surface, distance: float, speed: float, total_power: float):
    font = pygame.font.SysFont('Arial', 18, bold=True)
    small_font = pygame.font.SysFont('Arial', 12)
    panel_rect = pygame.Rect(10, 80, 130, 85)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, panel_rect, 2, border_radius=8)

    labels = [("Distance:", f"{distance:.1f}m"), ("Speed:", f"{speed * 3.6:.1f} km/h"), ("Power:", f"{total_power:.1f}W")]
    y_offset = 88
    for label, value in labels:
        surface.blit(small_font.render(label, True, (180, 180, 180)), (20, y_offset))
        surface.blit(font.render(value, True, (255, 255, 255)), (20, y_offset + 12))
        y_offset += 28


def draw_parallel_info(surface):
    font = pygame.font.SysFont('Arial', 12)
    panel_rect = pygame.Rect(10, 170, 130, 55)
    pygame.draw.rect(surface, PANEL_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, SLIDER_BORDER_COLOR, panel_rect, 2, border_radius=8)
    y_offset = 178
    for text in [f"Cores: {NUM_CORES}", f"Cats/Core: {CATS_PER_CORE}", f"Total Cats: {NUM_CATS}"]:
        surface.blit(font.render(text, True, (200, 200, 200)), (20, y_offset))
        y_offset += 15


def main():
    mp.set_start_method('spawn', force=True)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption(f"Cat Walking Simulation - {NUM_CORES} cores, {CATS_PER_CORE} cats/core")
    clock = pygame.time.Clock()

    spawn_x, ground_y = 10.0, 5.0

    # Shared memory for all cats
    shared_data = Array(ctypes.c_double, NUM_CATS * CAT_DATA_SIZE)
    running_flag = Value(ctypes.c_int, 1)
    reset_flags = [Value(ctypes.c_int, 0) for _ in range(NUM_CORES)]

    # Start workers
    workers = []
    for core_id in range(NUM_CORES):
        worker = Process(
            target=simulation_worker,
            args=(core_id, CATS_PER_CORE, shared_data, running_flag, reset_flags[core_id], spawn_x, ground_y)
        )
        worker.start()
        workers.append(worker)

    # Force history
    force_max_samples = int(5.0 * TARGET_FPS)
    force_histories = [collections.deque(maxlen=force_max_samples) for _ in range(8)]
    last_best_idx = -1

    iteration = 1
    iteration_time = 0.0
    running = True

    while running:
        dt = 1.0 / TARGET_FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    iteration_time = 0.0
                    for flag in reset_flags:
                        flag.value = 1
                    for h in force_histories:
                        h.clear()

        iteration_time += dt

        # Read all cat data
        all_cats = [read_cat_data(shared_data, i) for i in range(NUM_CATS)]

        # Find best cat
        best_idx = max(range(NUM_CATS), key=lambda i: all_cats[i]['distance'])
        best_cat = all_cats[best_idx]

        # Update force history
        if best_idx != last_best_idx:
            for h in force_histories:
                h.clear()
            last_best_idx = best_idx

        for i, muscle in enumerate(best_cat['muscle_data']):
            if i < len(force_histories):
                force_histories[i].append(muscle['activation'] * 1000)  # Approximate force

        camera_x = best_cat['body_position'][0]

        # Check iteration end
        if iteration_time >= ITERATION_TIME:
            iteration += 1
            iteration_time = 0.0
            for flag in reset_flags:
                flag.value = 1
            for h in force_histories:
                h.clear()

        # Render
        screen.fill(SKY_COLOR)
        ground_screen_y = SCREEN_HEIGHT - ground_y * PPM
        pygame.draw.rect(screen, GROUND_COLOR, (0, ground_screen_y, SCREEN_WIDTH, SCREEN_HEIGHT - ground_screen_y))

        draw_grass(screen, camera_x, ground_y)
        draw_ruler(screen, camera_x, ground_y)
        draw_flag(screen, spawn_x, camera_x, ground_y)

        for i, cat_data in enumerate(all_cats):
            draw_cat_from_render_data(screen, cat_data, camera_x)

        draw_iteration_info(screen, iteration, iteration_time, ITERATION_TIME)
        draw_parallel_info(screen)

        total_power = sum(m['activation'] * 1000 for m in best_cat['muscle_data'])
        draw_best_cat_info(screen, best_cat['distance'], best_cat['speed'], total_power)
        draw_muscle_energy_sliders(screen, best_cat['muscle_energies'], SCREEN_WIDTH - 260, 20)
        draw_force_graph(screen, force_histories, SCREEN_WIDTH - 270, 200, force_max_samples)

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    # Cleanup
    running_flag.value = 0
    for worker in workers:
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.terminate()

    pygame.quit()


if __name__ == "__main__":
    main()