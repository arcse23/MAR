import math
from queue import PriorityQueue
import numpy
from matplotlib import pyplot, patches

class Node:

    def __init__(self, x=None, y=None, cost=None, parent=None):
        self.position = [x, y]
        self.cost = cost
        self.parent = parent
    
    def update_cost(self, cost):
        self.cost = cost
    
    def update_parent(self, parent):
        self.parent = parent
    
    def __lt__(self, other):
        return self.cost < other.cost

class Grid:

    def __init__(self, size=None):
        self.N = size
        self.B = None
        self.obstacles = None
        self.s_index = None
        self.g_index = None 

    def random_endpoints(self):        
        while True:
            x1, y1 = numpy.random.randint(0, self.N, size=2)  
            x2, y2 = numpy.random.randint(0, self.N, size=2)  
            if math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)) >= 1.2*self.N:
                s_index = [x1, y1]
                g_index = [x2, y2]
                break
        return s_index, g_index
    
    def initialize_endpoints(self, s_index, g_index):
        self.s_index = s_index
        self.g_index = g_index

    def read_occupancy_grid(self, grid):
        self.B = numpy.copy(grid)
    
    def initialize_occupancy_grid(self):
        self.B = numpy.zeros([self.N, self.N], dtype=int)
    
    def update_occupancy_grid(self, obstacle):
        self.B[self.s_index[0]][self.s_index[1]] = 2
        self.B[self.g_index[0]][self.g_index[1]] = 4
        if obstacle is not None:
            ((x, y), w, h) = obstacle
            for i in range(x, x+w):
                for j in range(y, y+h):
                    self.B[i, j] = 1
    
    def print_occupancy_grid(self):
        for i in range(self.N):
            for j in range(self.N):
                print(self.B[i, j], end=" ")
            print()
    
    def create_obstacles(self, obstacle_count):
        self.update_occupancy_grid(None)
        self.obstacles = []
        count = 0
        k = self.N//100
        while count < obstacle_count:
            x, y = numpy.random.randint(1, self.N, size=2)
            w, h = k * numpy.random.randint(5, 11, size=2)
            if x+w < self.N and y+h < self.N:
                p = self.B[x][y]
                q = self.B[x+w-1][y]
                r = self.B[x][y+h-1]
                s = self.B[x+w-1][y+h-1]
                if p == 0 and q == 0 and r == 0 and s == 0:
                    self.obstacles.append(((x, y), w, h))
                    self.update_occupancy_grid(((x, y), w, h))
                    count += 1        
    
    def print_obstacles(self):
        print(self.obstacles)

class AStar:

    def __init__(self, grid=None):
        self.N = grid.N
        self.B = numpy.copy(grid.B)
        self.obstacles = grid.obstacles
        self.s_index = grid.s_index
        self.g_index = grid.g_index
        self.start = None
        self.goal = None
        self.path = None
    
    def find_8neighbours(self, position):
        neighbours_positions = []
        directions = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]]
        for dir in directions:
            x = position[0] + dir[0]
            y = position[1] + dir[1]
            if x >= 0 and y >= 0 and x < self.N and y < self.N:
                if self.B[x][y] != 1:
                    neighbours_positions.append([x, y])
        return neighbours_positions

    def create_node_grid(self):
        self.G = []
        for i in range(self.N):
            self.G.append([])
            for j in range(self.N):                
                if self.B[i][j] == 1:
                    self.G[i].append(None)
                else:
                    self.G[i].append(Node(i, j, float("inf"), None))
    
    def update_node_grid(self):
        x, y = self.s_index
        self.start = self.G[x][y]
        self.start.update_cost(0)
    
    def c1(self, node1, node2):
        x1, y1 = node1.position
        x2, y2 = node2.position
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        d = math.sqrt(dx*dx + dy*dy)
        return d    

    def h_euclidean(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        d = math.sqrt(dx*dx + dy*dy)
        return d

    def a_star(self):
        self.create_node_grid()
        self.update_node_grid()
        
        search_tree = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        search_tree.put((0, self.start))
        came_from[self.start] = None
        cost_so_far[self.start] = 0

        while not search_tree.empty():
            current = search_tree.get()[1]
            if current.position == self.g_index:
                self.goal = current
                break
            neighbour_positions = self.find_8neighbours(current.position)
            for [x, y] in neighbour_positions:
                next = self.G[x][y]
                #print(current.position, next.position)
                new_cost = cost_so_far[current] + self.c1(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:                    
                    next.update_cost(new_cost)
                    next.update_parent(current)
                    f_score = new_cost + self.h_euclidean(next.position, self.g_index)
                    search_tree.put((f_score, next))
                    came_from[next] = current
                    cost_so_far[next] = new_cost
        
        self.reconstruct_path(came_from)
        
    def reconstruct_path(self, came_from):
        if self.goal is not None:
            self.path = []
            current = self.goal
            while current is not None:
                self.path.append(current.position)
                x, y = current.position
                self.B[x][y] = 3
                current = came_from[current]
            self.path.reverse()
    
    def print_path(self):
        if self.path is not None:
            print("Cost of Path: ", self.goal.cost)
            print("Path: ", self.path)            
        else:
            print("No path to goal was found.")
    
    def display(self, ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)        
        if self.obstacles is not None:
            for ((x, y), w, h) in self.obstacles:
                    obstacle = patches.Rectangle((x, y), w, h, facecolor='black')
                    ax.add_patch(obstacle)
        x, y = [], []
        if self.path is not None:            
            for [i, j] in self.path:
                x.append(i)
                y.append(j)
        ax.plot(x, y, lw=1, mew=0, color='green')
        x, y = self.g_index
        ax.plot(x, y, marker='+', color='red')

def main():
    grid = Grid(1000)
    start, goal = grid.random_endpoints()
    grid.initialize_endpoints(start, goal)
    grid.initialize_occupancy_grid()
    grid.create_obstacles(20)
    #grid.print_occupancy_grid()

    fig, ax = pyplot.subplots()
    ax.set_title("A* Algorithm Path")

    A = AStar(grid)
    A.a_star()
    #A.print_path()
    A.display(ax)

    pyplot.show()

main()
