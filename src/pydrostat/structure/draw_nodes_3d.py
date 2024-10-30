import OpenGL.GL as gl
import OpenGL.GLU as glu
import pygame
import sys
import time


class NodeDrawer3D:
    def __init__(self, structure, dt=0.02):
        self.structure = structure
        self.dt = dt

        # pygame.init()
        # omit mixer init b/c sound not used and takes long time to load if
        # there are no outputs
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

        # pygame control parameters
        self.orbiting = False
        self.panning = False
        self.control_mod = False

    def main_loop(self, simulating=True):
        while self.running:
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
                        self.structure.iterate(self.dt)

                    # if event.key == pygame.K_LEFT:
                    #     self.structure.iterate(-self.dt)
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
                        # gl.glRotatef(event.rel[1], 1, 0, 0)
                    if self.panning:
                        gl.glTranslatef(0, 0, -event.rel[1] / 200)
                if event.type == pygame.MOUSEWHEEL:
                    gl.glTranslatef(0.0, event.y, 0.0)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            if self.structure.environment is not None:
                for food in self.structure.environment.food_locs:
                    gl.glPointSize(10.0)
                    gl.glColor3f(0, 1, 1)
                    gl.glBegin(gl.GL_POINTS)
                    gl.glVertex3fv(food)
                    gl.glEnd()

            sim_start = time.perf_counter()
            if simulating:
                self.structure.iterate(self.dt)
            sim_end = time.perf_counter()
            self.draw_structure()

            self.draw_obstacles()

            self.draw_axes()

            pygame.display.flip()
            self.clock.tick(self.fps)

            actual_fps = self.clock.get_fps()
            loop_period = (time.perf_counter() - loop_start) * 1000
            print(
                f"Actual FPS: {actual_fps:.2f} | Loop Period (ms): {loop_period:.2f} | Sim Time (ms): {(sim_end - sim_start)*1000:.2f}"
            )

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

    def draw_structure(self):
        gl.glBegin(gl.GL_LINES)
        for edge, muscle in zip(self.structure.edges, self.structure.muscles):
            activation = muscle / (1 + muscle)
            gl.glColor3f(1, 1 - activation, 1 - activation)
            for vertex in edge:
                gl.glVertex3fv(self.structure.positions[vertex])
        gl.glEnd()

        gl.glPointSize(10.0)
        gl.glBegin(gl.GL_POINTS)
        scents = self.structure.smell(self.structure.positions)
        max_scent = max(scents)
        for vertex, scent in zip(self.structure.positions, scents):
            gl.glColor3f(1, 1 - scent / max_scent, 1 - scent / max_scent)
            gl.glVertex3fv(vertex)
        # for vertex in self.structure.positions:
        #     gl.glVertex3fv(vertex)
        gl.glEnd()

    def draw_obstacles(self):
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1, 0, 1)
        for obstacle in self.structure.obstacles:
            for edge in obstacle.edges:
                for vertex in edge:
                    gl.glVertex3fv(obstacle.vertices[vertex])
        gl.glEnd()
