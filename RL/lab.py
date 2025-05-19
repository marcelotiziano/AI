import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import collections
import time

class InteractiveMaze:
    def __init__(self, width=6, height=6):
        self.base_width = width
        self.base_height = height
        self.reset_maze()

    def reset_maze(self):
        self.width = self.base_width
        self.height = self.base_height
        self.start = (0, 0)
        self.goal = (self.height - 1, self.width - 1)
        self._generate_traps()
        self.reset_agent()

    def _generate_traps(self):
        num_traps = max(3, (self.width * self.height) // 10)
        cells = [(r, c) for r in range(self.height) for c in range(self.width)
                 if (r, c) not in [self.start, self.goal]]
        choices = np.random.choice(len(cells), num_traps, replace=False)
        self.traps = {cells[i] for i in choices}

    def increase_complexity(self):
        self.base_width += 1
        self.base_height += 1
        self.reset_maze()

    def reset_agent(self):
        self.agent_pos = self.start

    def draw_maze(self, ax):
        grid = np.zeros((self.height, self.width))
        for (r, c) in self.traps:
            grid[r, c] = -1
        grid[self.goal] = 2
        ax.clear()
        ax.imshow(grid, cmap='viridis', vmin=-1, vmax=2)
        ax.scatter(self.agent_pos[1], self.agent_pos[0], c='red', s=100, label='Agent')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Maze {self.width}×{self.height}')
        ax.legend(loc='upper right')

    def find_path_bfs(self):
        queue = collections.deque([(self.start, [])])
        visited = {self.start}
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.goal:
                return path
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    new = (nr, nc)
                    if new not in visited and new not in self.traps:
                        visited.add(new)
                        queue.append((new, path + [new]))
        return []

    def compute_reward(self, pos, visited):
        r = -0.1
        if pos in self.traps:
            r -= 5
        if pos == self.goal:
            r += 10
        if pos not in visited and pos != self.goal:
            r += 0.5
        return r

# Initialize environment and figure
maze = InteractiveMaze()
fig, (ax_maze, ax_table) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Initial draw
maze.draw_maze(ax_maze)
ax_table.axis('off')
ax_table.set_title('Reward Table (per move)')

# Buttons
ax_refresh = plt.axes([0.1, 0.1, 0.2, 0.075])
btn_refresh = Button(ax_refresh, 'Refresh Maze')
ax_complex = plt.axes([0.4, 0.1, 0.2, 0.075])
btn_complex = Button(ax_complex, 'Increase Complexity')
ax_start = plt.axes([0.7, 0.1, 0.2, 0.075])
btn_start = Button(ax_start, 'Start Agent')

def refresh(event):
    maze.reset_maze()
    maze.draw_maze(ax_maze)
    ax_table.clear()
    ax_table.axis('off')
    ax_table.set_title('Reward Table (per move)')
    fig.canvas.draw_idle()

def increase(event):
    maze.increase_complexity()
    maze.draw_maze(ax_maze)
    ax_table.clear()
    ax_table.axis('off')
    ax_table.set_title('Reward Table (per move)')
    fig.canvas.draw_idle()

def start(event):
    path = maze.find_path_bfs()
    if not path:
        print("No path found!")
        return
    maze.reset_agent()
    visited = {maze.start}
    table_data = [["Step", "Position", "Reward", "Cumulative"]]
    cum_reward = 0

    # Animate each move
    for step, pos in enumerate(path, start=1):
        # Compute reward
        reward = maze.compute_reward(pos, visited)
        visited.add(pos)
        cum_reward += reward

        # Update agent position and maze
        maze.agent_pos = pos
        maze.draw_maze(ax_maze)

        # Update reward table
        ax_table.clear()
        ax_table.axis('off')
        ax_table.set_title('Reward Table (per move)')
        table_data.append([step, str(pos), f"{reward:.2f}", f"{cum_reward:.2f}"])
        tbl = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)

        # Draw and pause
        fig.canvas.draw_idle()
        plt.pause(1)

btn_refresh.on_clicked(refresh)
btn_complex.on_clicked(increase)
btn_start.on_clicked(start)

plt.show()
