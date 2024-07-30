import numpy as np
import datetime


class DataLogger:
    def __init__(self, edges):
        self.edges = edges

        self.timestamps = []
        self.pos = []
        self.vel = []
        self.acc = []
        self.ext_forces = []
        self.int_forces = []

    def log(self, timestamp, pos, vel, acc, ext_force, int_force):
        self.timestamps.append(timestamp)
        self.pos.append(pos.copy())
        self.vel.append(vel.copy())
        self.acc.append(acc)
        self.ext_forces.append(ext_force.copy())
        self.int_forces.append(int_force.copy())

    def get_data(self):
        return {
            "timestamps": self.timestamps,
            "pos": self.pos,
            "vel": self.vel,
            "acc": self.acc,
            "ext_forces": self.ext_forces,
            "int_forces": self.int_forces,
        }

    def save(self, filename: str = None):
        if filename is None:
            filename = datetime.datetime.now().strftime("simulation_%Y%m%d_%H%M%S.npz")

        np.savez(
            filename,
            edges=self.edges,
            timestamps=self.timestamps,
            pos=self.pos,
            vel=self.vel,
            acc=self.acc,
            ext_forces=self.ext_forces,
            int_forces=self.int_forces,
        )

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        cls.edges = data["edges"]
        cls.timestamps = data["timestamps"]
        cls.pos = data["pos"]
        cls.vel = data["vel"]
        cls.acc = data["acc"]
        cls.ext_forces = data["ext_forces"]
        cls.int_forces = data["int_forces"]
