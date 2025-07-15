# TODO: render and simulation should be separated
import OpenGL.GL as gl
import OpenGL.GLU as glu
from PIL import Image
import pygame
import sys
import time

import numpy as np

from pydrostat.actor import Actor
from pydrostat.environment.environment import Environment


class Scene:
    def __init__(self, actors: list[Actor], environment: Environment, dt=0.02):
        self.actors = actors
        self.environment = environment
        for actor in self.actors:
            actor.set_environment(environment)
        self.dt = dt

        pygame.display.init()
        pygame.joystick.init()

        display = (800, 600)
        self.screen = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
        self.fps = 1 / self.dt
        self.clock = pygame.time.Clock()
        self.running = True

        glu.gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
        gl.glTranslatef(0.0, -3, -10)
        gl.glRotatef(-70, 1, 0, 0)
        gl.glTranslatef(0.0, 3, 0)

        # pygame control parameters
        self.orbiting = False
        self.panning = False
        self.control_mod = False
        self.frame_count = 0

    def main_loop(self, simulating=False, rotating=False, recording=False):
        while self.running:
            if rotating:
                gl.glRotatef(0.2, 0, 0, 1)

            loop_start = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LCTRL:
                        self.control_mod = True
                        if self.orbiting:
                            self.orbiting = False
                            self.panning = True
                    if event.key == pygame.K_SPACE:
                        simulating = not simulating

                    if event.key == pygame.K_RIGHT:
                        for actor in self.actors:
                            actor.step(self.frame_count * self.dt, self.dt)

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LCTRL:
                        self.control_mod = False
                        if self.panning:
                            self.orbiting = True
                            self.panning = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.control_mod:
                        if event.button == 2:
                            self.panning = True
                    else:
                        if event.button == 2 or event.button == 3:
                            self.orbiting = True
                if event.type == pygame.MOUSEBUTTONUP:
                    if self.control_mod:
                        if event.button == 2:
                            self.panning = False
                    else:
                        if event.button == 2 or event.button == 3:
                            self.orbiting = False
                if event.type == pygame.MOUSEMOTION:
                    if self.orbiting:
                        gl.glRotatef(event.rel[0], 0, 0, 1)
                    if self.panning:
                        gl.glTranslatef(0, 0, -event.rel[1] / 200)
                if event.type == pygame.MOUSEWHEEL:
                    gl.glTranslatef(0.0, event.y, 0.0)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            if self.environment.food_locations is not None:
                for food in self.environment.food_locations:
                    gl.glPointSize(10.0)
                    gl.glColor3f(0, 1, 1)
                    gl.glBegin(gl.GL_POINTS)
                    gl.glVertex3fv(food)
                    gl.glEnd()

            sim_start = time.perf_counter()
            if simulating:
                for actor in self.actors:
                    actor.step(self.frame_conut * self.dt, self.dt)
            sim_end = time.perf_counter()
            self.draw_actors()

            self.draw_obstacles()

            self.draw_axes()

            if recording:
                width, height = pygame.display.get_surface().get_size()
                pixels = gl.glReadPixels(
                    0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
                )
                image = Image.frombytes("RGB", (width, height), pixels)
                image = image.transpose(
                    Image.FLIP_TOP_BOTTOM
                )  # OpenGL origin is at the bottom-left
                image.save(f"render/frames/frame_{self.frame_count:04d}.png")

            pygame.display.flip()
            self.clock.tick(self.fps)

            actual_fps = self.clock.get_fps()
            loop_period = (time.perf_counter() - loop_start) * 1000
            print(
                f"Actual FPS: {actual_fps:.2f} | Loop Period (ms): {loop_period:.2f} | Sim Time (ms): {(sim_end - sim_start)*1000:.2f}"
            )
            self.frame_count += 1

    def draw_axes(self):
        gl.glBegin(gl.GL_LINES)

        gl.glColor3f(1, 0, 0)
        gl.glVertex3fv([0, 0, 0])
        gl.glVertex3fv([1, 0, 0])

        gl.glColor3f(0, 1, 0)
        gl.glVertex3fv([0, 0, 0])
        gl.glVertex3fv([0, 1, 0])

        gl.glColor3f(0, 0, 1)
        gl.glVertex3fv([0, 0, 0])
        gl.glVertex3fv([0, 0, 1])

        gl.glEnd()

    def draw_actors(self):
        for idx, actor in enumerate(self.actors):
            actor.draw(idx)

    # def draw_actors(self):
    #     color = np.array([0, 0, 0], dtype=float)
    #     for idx, actor in enumerate(self.actors):
    #         gl.glBegin(gl.GL_LINES)
    #         for edge, input in zip(actor.model.edges, structure.control_inputs):
    #             activation = input / (1 + input)
    #             color[:] = 1 - activation
    #             color[idx] = 1
    #             gl.glColor3f(*color)
    #             for vertex in edge:
    #                 gl.glVertex3f(*structure.positions[vertex])
    #         gl.glEnd()

    #         gl.glPointSize(10.0)
    #         gl.glBegin(gl.GL_POINTS)
    #         scents = self.environment.sample_scent(structure.positions)
    #         max_scent = max(scents)
    #         for vertex, scent in zip(structure.positions, scents):
    #             color[:] = 1 - scent / max_scent
    #             color[idx] = 1
    #             gl.glColor3f(*color)
    #             gl.glVertex3f(*vertex)
    #         gl.glEnd()

    def draw_obstacles(self):
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1, 0, 1)
        for obstacle in self.environment.obstacles:
            obstacle.draw()
        gl.glEnd()
