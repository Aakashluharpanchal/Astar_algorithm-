import numpy as np
import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current node to goal)
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Calculate the Manhattan distance between points a and b."""
    return abs(a.x - b.x) + abs(a.y - b.y)


def generate_random_map(width, height, obstacle_prob):
    """Generate a random map with obstacles."""
    map_grid = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if np.random.random() < obstacle_prob:
                map_grid[i, j] = 1
    return map_grid


def is_valid_node(map_grid, node):
    """Check if the node is within map bounds and not an obstacle."""
    if node.x < 0 or node.x >= map_grid.shape[1] or node.y < 0 or node.y >= map_grid.shape[0]:
        return False
    if map_grid[node.y, node.x] == 1:
        return False
    return True


def astar(map_grid, start, goal):
    """A* algorithm implementation."""
    open_list = []
    closed_list = set()

    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add((current_node.x, current_node.y))

        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_pos = (current_node.x + new_pos[0], current_node.y + new_pos[1])

            if not is_valid_node(map_grid, Node(node_pos[0], node_pos[1])):
                continue

            neighbor_node = Node(node_pos[0], node_pos[1], current_node)

            if (neighbor_node.x, neighbor_node.y) in closed_list:
                continue

            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor_node, goal_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if any(open_node.x == neighbor_node.x and open_node.y == neighbor_node.y and neighbor_node.g >= open_node.g for open_node in open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return None


def plot_path(map_grid, path, start, goal):
    plt.imshow(map_grid, cmap='Greys', origin='lower')
    plt.plot(start[0], start[1], 'ro', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    if path:
        for i in range(1, len(path)):
            plt.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], 'b-', linewidth=2)

    plt.title('A* Pathfinding')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def main():
    width = 50
    height = 50
    obstacle_prob = 0.2

    map_grid = generate_random_map(width, height, obstacle_prob)

    start = (5, 5)
    goal = (45, 45)

    path = astar(map_grid, start, goal)

    if path:
        print("Path found!")
        plot_path(map_grid, path, start, goal)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
