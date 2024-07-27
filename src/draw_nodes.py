import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class NodeDrawer:
    """Given a cell or arm, draw the network. Other features include
    - Click and drag vertices (callback to cell object for calculation)

    Args
        structure: a cell or arm of multiple cells. Must have
            Attribute:  vertices - Nx2 np.ndarray of vertex positions
                        edges - list of tuples of vertex indices that form edges
                        dof_matrix - NxD matrix of degrees of freedom for each vertex

            Methods:    move_vertex(index, abs_coord/rel_coord) - move a vertex to a new position
    """

    def __init__(self, structure):
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

        self.update_plot()

        self.dragging_node = None

    def on_press(self, event):
        """Handler for mouse button press event."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        click_coord = np.array([x, y])

        self.dragging_node = np.argmin(
            np.linalg.norm(self.vertices - click_coord[None, :], axis=1)
        )

    def on_release(self, event):
        """Handler for mouse button release event."""
        self.dragging_node = None

    def on_motion(self, event):
        """Handler for mouse motion event."""
        if self.dragging_node is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        updated_vertices = self.structure.move_vertex(
            self.dragging_node, abs_coord=np.array([x, y])
        )
        self.vertices = updated_vertices
        for i, new_vertex in enumerate(updated_vertices):
            self.pos[i] = tuple(new_vertex)
            self.graph.nodes[i]["pos"] = tuple(new_vertex)

        self.update_plot()

    def update_plot(self):
        """Update the plot with new node positions."""
        self.ax.clear()
        nx.draw_networkx(
            self.graph, self.pos, ax=self.ax, with_labels=False, node_size=10
        )
        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
