import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

from data_logger import DataLogger


class NodeDrawer:
    """Given a cell or arm, draw the network. Other features include
    - Click and drag vertices (callback to cell object for calculation)

    Args
        structure: a cell or arm of multiple cells. Must have
            Attribute:  vertices - Nx2 np.ndarray of vertex positions
                        edges - list of tuples of vertex indices that form edges

            Methods:    apply_force(index, force vector) - apply an external force
                            to a particular vertex
    """

    def __init__(self, structure, dt=0.05):
        self.dt = dt
        self.structure = structure
        self.vertices = structure.vertices
        self.edges = structure.edges

        # Setup graph
        self.graph = nx.Graph()
        for i, vertex in enumerate(self.vertices):
            self.graph.add_node(i, pos=vertex)
        self.graph.add_edges_from(self.edges)
        self.pos = nx.get_node_attributes(self.graph, "pos")

        # Setup plot interactivity
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.dragging_node = None  # vertex index that is being acted upon
        self.force_vec = None  # array with force components
        self.paused = False  # whether or not the animation/simulation is paused
        self.end_animation = False  # whether the simulation has been ended

        # Edit for Colab: 10s anim instead of infinite
        self.anim = FuncAnimation(
            self.fig,
            self.update_plot,
            # frames=self.infinite_frames,
            frames=int(10 / dt),
            interval=int(dt * 1000),
            blit=True,
            save_count=1000,
        )

    def infinite_frames(self):
        frame = 0
        while not self.end_animation:
            yield frame
            frame += 1

    def on_key_press(self, event):
        if event.key == " ":
            self.paused = not self.paused
        if event.key == "q":
            self.end_animation = True

        if self.paused:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()

        if self.end_animation:
            self.ani.event_source.stop()
            self.structure.save()
            self.save_sim_rerun(self.structure.logger)

    def on_press(self, event):
        """Handler for mouse button press event."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        click_coord = np.array([x, y])

        self.dragging_node = np.argmin(
            np.linalg.norm(self.vertices - click_coord[None, :], axis=1)
        )

    def on_release(self, _):
        """Handler for mouse button release event."""
        self.structure.apply_external_force(self.dragging_node, force=np.array([0, 0]))
        self.dragging_node = None
        self.force_vec = None

    def on_motion(self, event):
        """Handler for mouse motion event."""
        if self.dragging_node is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        self.force_vec = np.array([x, y]) - self.vertices[self.dragging_node]
        self.force_vec /= np.linalg.norm(self.force_vec)
        self.structure.apply_external_force(
            self.dragging_node,
            force=self.force_vec * 5,
        )

    def update_plot(self, _):
        """Update the plot with new node positions."""
        self.ax.clear()
        (_,) = self.ax.plot(0, 0)

        self.structure.calc_next_states(self.dt)
        self.vertices = self.structure.vertices
        for i, new_vertex in enumerate(self.vertices):
            self.pos[i] = tuple(new_vertex)
            self.graph.nodes[i]["pos"] = tuple(new_vertex)

        nx.draw(
            self.graph,
            self.pos,
            ax=self.ax,
            with_labels=False,
            node_size=10,
            edge_color=self.structure.muscles,
        )
        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-1, 15])
        self.ax.set_aspect("equal")
        if self.dragging_node is not None:
            self.ax.arrow(
                *self.vertices[self.dragging_node],
                *self.force_vec,
                head_width=0.1,
                head_length=0.1,
                fc="k",
                ec="k",
            )

        if self.structure.odor_func is not None:
            x = np.linspace(-10, 10, 100)
            y = np.linspace(-1, 15, 100)
            X, Y = np.meshgrid(x, y)
            z = self.structure.odor_func(X, Y)
            self.ax.contour(X, Y, z)

        return (_,)

    def save_sim_rerun(self, logger=None, filename=None):
        """Recreates a logged simulation and saves the animation."""
        if (logger is None) == (filename is None):
            raise ValueError("Either logger or filename must be provided, not both.")

        if logger is None:
            logger = DataLogger.load(filename)

        if filename is None:
            filename = "simulation.mp4"

        fig, ax = plt.subplots()

        graph = nx.Graph()
        for i, vertex in enumerate(logger.pos[0].reshape(-1, 2)):
            self.graph.add_node(i, pos=vertex)
        graph.add_edges_from(logger.edges)
        pos = nx.get_node_attributes(graph, "pos")

        def update(frame):
            ax.clear()
            for i, new_vertex in enumerate(logger.pos[frame].reshape(-1, 2)):
                pos[i] = tuple(new_vertex)
                graph.nodes[i]["pos"] = tuple(new_vertex)

            nx.draw(
                graph,
                pos,
                ax=ax,
                with_labels=False,
                node_size=10,
            )
            ax.set_axis_on()
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_xlim([-10, 10])
            ax.set_ylim([-1, 15])
            ax.set_aspect("equal")

            if any(logger.ext_forces[frame] != 0):
                dragging_node = np.where(logger.ext_forces[frame] != 0)[0][0]
                force_vec = logger.ext_forces[frame][dragging_node : dragging_node + 2]
                force_vec /= np.linalg.norm(force_vec)

                ax.arrow(
                    *logger.pos[frame][dragging_node : dragging_node + 2],
                    *force_vec,
                    head_width=0.1,
                    head_length=0.1,
                    fc="k",
                    ec="k",
                )

        print("Beggining reanimation")
        ani = FuncAnimation(
            fig, update, frames=len(logger.timestamps), interval=self.dt * 1000
        )
        ani.save(filename)
