import OpenGL.GL as gl
import OpenGL.GLU as glu
import pygame
import sys


class NodeDrawer3D:
    def __init__(self, structure, dt=0.02):
        self.structure = structure
        self.dt = dt

        pygame.init()

        display = (800, 600)
        self.screen = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
        self.fps = 1 / self.dt
        self.clock = pygame.time.Clock()
        self.running = True

        glu.gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        gl.glTranslatef(0.0, 0.0, -10)
        gl.glRotatef(20, 1, 0, 0)

    def main_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

            gl.glRotatef(1, 0, 1, 0)  # orbit view
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            self.draw_structure()

            actual_fps = self.clock.get_fps()
            print("Actual FPS:", actual_fps)

            pygame.display.flip()
            self.clock.tick(self.fps)

    def draw_structure(self):
        self.structure.calc_next_states(self.dt)
        gl.glBegin(gl.GL_LINES)
        for edge in self.structure.edges:
            for vertex in edge:
                gl.glVertex3fv(self.structure.vertices[vertex])
        gl.glEnd()


if __name__ == "__main__":
    drawer = NodeDrawer3D(None, 1 / 60)
    drawer.main_loop()


# if __name__ == "__main__":
#     pygame.init()

#     display = (800, 600)
#     screen = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
#     fps = 60
#     dt = 1 / 60

#     clock = pygame.time.Clock()
#     running = True

#     glu.gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
#     gl.glTranslatef(0.0, 0.0, -10)
#     gl.glRotatef(20, 1, 0, 0)

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 pygame.quit()
#                 quit()

#         gl.glRotatef(1, 0, 1, 0)  # orbit view
#         gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

#         # draw lines
#         fps = clock.get_fps()
#         print(fps)

#         pygame.display.flip()
#         clock.tick(60)
