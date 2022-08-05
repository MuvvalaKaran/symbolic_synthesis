import graphviz as gv
from cudd import Cudd


class GridWorld:
    def __init__(self, size: int):
        self.grid_size = size

    def build_grid_world(self):
        interior = set()
        que = [(0, 0)]
        grid_blocks = [[(i, j) for i in range(self.grid_size)] for j in range(self.grid_size)]
        # while que:
        #     posn = que.pop()
        #     if not posn in interior:
        #         interior.add(posn)
        #         i, j = posn
        #         if i == 0 or j == 0:
        #             raise ValueError("Unbounded warehouse!")
        #         if i + 1 == len(level_lines) or j + 1 == len(level_lines[i]):
        #             raise ValueError("Unblunded warehouse!")
        #         if grid_blocks[i][j - 1] == '#':
        #             que.append((i, j - 1))
        #         if grid_blocks[i - 1][j] == '#':
        #             que.append((i - 1, j))
        #         if grid_blocks[i][j + 1] != '#':
        #             que.append((i, j + 1))
        #         if grid_blocks[i + 1][j] != '#':
        #             que.append((i + 1, j))
        # # Get list of interior positions sorted by rows and columns.
        # pairs = list(interior)
        grid_blocks.sort()
        return grid_blocks




if __name__ == "__main__":
    grid_world = GridWorld(size=3)
    grid_world.build_grid_world()