import tkinter as tk
import numpy as np
import time

def hex_to_rgb(hex_color):
    """Convert a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    """Convert an (R, G, B) tuple to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)

def blend_with_white(hex_color, mix=0.7):
    """
    Return a lighter shade of hex_color by blending with white.
    mix: Fraction of white (0 means original color, 1 means completely white).
    """
    rgb = hex_to_rgb(hex_color)
    blended_rgb = tuple(round((1 - mix) * c + mix * 255) for c in rgb)
    return rgb_to_hex(blended_rgb)

def darken_color(hex_color, mix=0.3):
    """
    Return a darker version of hex_color by blending with black.
    mix: Fraction of black (0 means original color, 1 means completely black).
    """
    rgb = hex_to_rgb(hex_color)
    dark_rgb = tuple(round((1 - mix) * c) for c in rgb)
    return rgb_to_hex(dark_rgb)

class MultiAgentPathVisualizer(tk.Tk):
    def __init__(self, env, trajectories, agent_positions, goal_positions,
                 cell_size=None, frames_per_transition=10, delay=50):
        """
        Parameters:
            env: The GridEnvironment instance (with a 2D numpy array attribute 'grid').
            trajectories: List of lists (one per agent) of (row, col) tuples.
            agent_positions: List of starting positions for each agent.
            goal_positions: List of goal positions for each agent.
            cell_size: Pixel size for each grid cell. If None, auto-sized based on grid and screen size.
            frames_per_transition: Number of intermediate frames between discrete cells.
            delay: Delay (milliseconds) between frames.
        """
        super().__init__()
        self.title("Multi-Agent Path Planning Visualization")

        # Store environment and trajectory data.
        self.env = env
        self.trajectories = trajectories
        self.agent_positions = agent_positions  # Not used directly (trajectories hold full history).
        self.goal_positions = goal_positions
        self.frames_per_transition = frames_per_transition
        self.delay = delay

        self.rows, self.cols = self.env.grid.shape

        # Auto-calculate cell_size based on available screen space if not provided
        if cell_size is None:
            self.cell_size = self._calculate_optimal_cell_size()
        else:
            self.cell_size = cell_size

        self.canvas_width = self.cols * self.cell_size
        self.canvas_height = self.rows * self.cell_size

        # Define fixed info panel width.
        info_panel_width = 300
        total_width = info_panel_width + self.canvas_width
        control_panel_height = 75  # Adjust to the height needed for your slider and buttons.
        total_height = self.canvas_height + control_panel_height
        self.geometry(f"{total_width}x{total_height}+100+100")

        self.configure(bg="#F5F5F5")
        
        # Define colors.
        self.agent_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        self.goal_colors = [blend_with_white(c, mix=0.7) for c in self.agent_colors]
        self.agent_dark_colors = [darken_color(c, mix=0.3) for c in self.agent_colors]
        
        # Create main layout frame.
        self.main_frame = tk.Frame(self, bg="#F5F5F5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left info panel.
        self.info_frame = tk.Frame(self.main_frame, bg="#F5F5F5", width=info_panel_width)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.info_frame.pack_propagate(False)
        
        # Info panel labels.
        self.time_label = tk.Label(self.info_frame, text="", font=("Helvetica", 14, "bold"),
                                   fg="black", bg="#F5F5F5")
        self.time_label.pack(anchor="nw", pady=(0,10))
        self.agent_count_label = tk.Label(self.info_frame, text="", font=("Helvetica", 14, "bold"),
                                          fg="black", bg="#F5F5F5")
        self.agent_count_label.pack(anchor="nw", pady=(0,10))
        
        # Collision analysis panel.
        self.conflict_label = tk.Label(self.info_frame, text="", font=("Helvetica", 12, "bold"),
                                       fg="green", bg="#F5F5F5", anchor="nw", justify=tk.LEFT, wraplength=280)
        self.conflict_label.pack(anchor="nw", pady=(0,10))

        # Collision details frame (scrollable area for conflict details)
        self.conflict_details_frame = tk.Frame(self.info_frame, bg="#F5F5F5")
        self.conflict_details_frame.pack(anchor="nw", pady=(0,10), fill=tk.BOTH, expand=True)
        self.conflict_detail_labels = []
        
        self.throughput_label = tk.Label(self.info_frame, text="", font=("Helvetica", 14, "bold"),
                                         fg="black", bg="#F5F5F5")
        self.throughput_label.pack(anchor="nw", pady=(0,10))

        # Add toggle for path tick marks
        self.show_tick_marks = tk.BooleanVar(value=False)
        self.tick_marks_checkbox = tk.Checkbutton(
            self.info_frame,
            text="Show Path Tick Marks",
            variable=self.show_tick_marks,
            font=("Helvetica", 12),
            fg="black",
            bg="#F5F5F5",
            activebackground="#F5F5F5",
            selectcolor="white",
            command=self.redraw
        )
        self.tick_marks_checkbox.pack(anchor="nw", pady=(0,10))

        # Right panel: Canvas.
        self.canvas_frame = tk.Frame(self.main_frame, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Below the canvas, create a control frame for discrete time step navigation.
        # Pack this FIRST so it reserves space before the canvas expands
        self.controls_frame = tk.Frame(self.canvas_frame, bg="white")
        self.controls_frame.pack(fill=tk.X, pady=(5,0), side=tk.BOTTOM)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width,
                                height=self.canvas_height, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind canvas resize event to update visualization
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # Create previous, pause/resume, and next buttons.
        self.prev_button = tk.Button(self.controls_frame, text="<", font=("Helvetica", 14, "bold"),
                                     fg="black", bg="white", relief=tk.RAISED, bd=2,
                                     command=self.step_back)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = tk.Button(self.controls_frame, text="⏸", font=("Helvetica", 14, "bold"),
                                      fg="black", bg="white", relief=tk.RAISED, bd=2,
                                      command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = tk.Button(self.controls_frame, text=">", font=("Helvetica", 14, "bold"),
                                     fg="black", bg="white", relief=tk.RAISED, bd=2,
                                     command=self.step_forward)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Create a slider for discrete time steps below the buttons.
        # Slider range: 0 to (max_discrete - 1)
        self.max_discrete = max(len(traj) for traj in self.trajectories if traj)
        self.time_slider = tk.Scale(self.controls_frame, from_=0, to=self.max_discrete - 1,
                                    orient=tk.HORIZONTAL, length=self.canvas_width - 100,
                                    bg="white", fg="black", troughcolor="white",
                                    font=("Helvetica", 14), command=self.slider_update)
        self.time_slider.pack(side=tk.LEFT, padx=5)
        
        # Animation state.
        self.current_frame = 0
        self.total_frames = (self.max_discrete - 1) * self.frames_per_transition + 1
        
        # Throughput (FPS) tracking.
        self.last_update_time = time.time()
        self.fps = 0
        
        self.paused = False
        self.after(0, self.update_frame)

    def _calculate_optimal_cell_size(self):
        """
        Calculate optimal cell size based on grid dimensions and available screen space.
        Ensures the visualization fits nicely on the screen.
        """
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Leave margins and account for info panel and control panel
        info_panel_width = 300
        margins = 100
        control_panel_height = 150

        # Available space for canvas
        available_width = screen_width - info_panel_width - margins
        available_height = screen_height - control_panel_height - margins

        # Calculate cell sizes that would fit in available space
        cell_size_width = int(available_width / self.cols)
        cell_size_height = int(available_height / self.rows)

        # Use the smaller of the two to maintain square cells and fit in available space
        optimal_cell_size = min(cell_size_width, cell_size_height)

        # Enforce reasonable bounds
        optimal_cell_size = max(optimal_cell_size, 5)    # Minimum 5 pixels per cell
        optimal_cell_size = min(optimal_cell_size, 100)  # Maximum 100 pixels per cell

        return optimal_cell_size

    def on_canvas_resize(self, event):
        """
        Handle canvas resize events to dynamically adjust cell size and redraw.
        """
        # Calculate new cell size based on actual canvas size
        new_cell_size_width = event.width / self.cols
        new_cell_size_height = event.height / self.rows

        new_cell_size = min(new_cell_size_width, new_cell_size_height)

        # Only redraw if cell size changed significantly (to avoid excessive redraws)
        if abs(new_cell_size - self.cell_size) > 1:
            self.cell_size = max(int(new_cell_size), 5)
            self.redraw()

    def detect_collisions_at_timestep(self, timestep):
        """
        Detect collisions at a specific timestep.
        Returns: list of dicts with collision info {agents: [id1, id2], position: (r,c)}
        """
        collisions = []
        position_map = {}  # Maps position to list of agent indices at that position

        for agent_idx, traj in enumerate(self.trajectories):
            if not traj:
                continue
            # Get agent position at this timestep
            if timestep >= len(traj):
                pos = tuple(traj[-1])
            else:
                pos = tuple(traj[timestep])

            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(agent_idx)

        # Find positions with multiple agents
        for pos, agents in position_map.items():
            if len(agents) > 1:
                collisions.append({
                    'agents': [a + 1 for a in agents],  # Convert to 1-indexed
                    'position': pos
                })

        return collisions

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="▶" if self.paused else "⏸")
        if not self.paused:
            self.after(self.delay, self.update_frame)
    
    def step_back(self):
        # Go back one discrete step.
        current_discrete = self.current_frame // self.frames_per_transition
        new_discrete = max(current_discrete - 1, 0)
        self.current_frame = new_discrete * self.frames_per_transition
        self.time_slider.set(new_discrete)
        self.redraw()
    
    def step_forward(self):
        # Advance one discrete step.
        current_discrete = self.current_frame // self.frames_per_transition
        new_discrete = min(current_discrete + 1, self.max_discrete - 1)
        self.current_frame = new_discrete * self.frames_per_transition
        self.time_slider.set(new_discrete)
        self.redraw()
    
    def slider_update(self, value):
        # When the slider is moved manually.
        discrete_val = int(value)
        self.current_frame = discrete_val * self.frames_per_transition
        self.redraw()
    
    def redraw(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_obstacles()
        if self.show_tick_marks.get():
            self.draw_tick_marks()
        self.draw_goal_circles()
        self.update_metrics()
        self.update_idletasks()
        self.draw_agents()

    
    def update_metrics(self):
        current_discrete = min(self.current_frame // self.frames_per_transition + 1, self.max_discrete)
        self.time_label.config(text=f"Time Step: {current_discrete}/{self.max_discrete}")
        n_agents = len(self.trajectories)
        self.agent_count_label.config(text=f"Agents: {n_agents}")

        # Detect collisions at current timestep
        timestep = self.current_frame // self.frames_per_transition
        collisions = self.detect_collisions_at_timestep(timestep)

        # Clear previous conflict detail labels
        for lbl in self.conflict_detail_labels:
            lbl.destroy()
        self.conflict_detail_labels = []

        # Display collision status
        if not collisions:
            self.conflict_label.config(text="✓ No Conflicts", fg="green")
        else:
            self.conflict_label.config(text=f"⚠ {len(collisions)} Conflict(s)", fg="red")
            # Display each conflict
            for collision in collisions:
                agent_ids = ", ".join(str(a) for a in collision['agents'])
                pos = collision['position']
                detail_text = f"Agents {agent_ids}\nat ({pos[0]}, {pos[1]})"
                lbl = tk.Label(self.conflict_details_frame, text=detail_text,
                              font=("Helvetica", 10), fg="red", bg="#F5F5F5",
                              anchor="nw", justify=tk.LEFT)
                lbl.pack(anchor="nw", pady=2)
                self.conflict_detail_labels.append(lbl)

        now = time.time()
        dt = now - self.last_update_time
        self.fps = 1/dt if dt > 0 else self.fps
        self.last_update_time = now
        self.throughput_label.config(text=f"FPS: {self.fps:.1f}")
    
    def update_frame(self):
        if not self.paused:
            self.redraw()
            self.current_frame += 1
            current_discrete = self.current_frame // self.frames_per_transition
            self.time_slider.set(current_discrete)
        if self.current_frame < self.total_frames and not self.paused:
            self.after(self.delay, self.update_frame)
    
    def draw_grid(self):
        for i in range(self.rows):
            for j in range(self.cols):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray", width=1)
    
    def draw_obstacles(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.env.grid[i, j] == -1:
                    x1, y1 = j * self.cell_size, i * self.cell_size
                    x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#444444", outline="#333333")
    
    def draw_tick_marks(self):
        """Draw tick marks showing the path history for each agent."""
        for idx, traj in enumerate(self.trajectories):
            if not traj:
                continue
            seg_index = self.current_frame // self.frames_per_transition
            agent_color = self.agent_colors[idx % len(self.agent_colors)]

            # Draw tick marks for visited positions
            for i, point in enumerate(traj[:seg_index+1]):
                col, row = point[1], point[0]
                x = (col + 0.5) * self.cell_size
                y = (row + 0.5) * self.cell_size
                tick_size = self.cell_size * 0.15

                # Draw small tick mark (cross or circle)
                self.canvas.create_oval(
                    x - tick_size, y - tick_size,
                    x + tick_size, y + tick_size,
                    fill=agent_color, outline=agent_color, width=1
                )

    def draw_goal_circles(self):
        for idx, goal in enumerate(self.goal_positions):
            traj = self.trajectories[idx]
            reached = (self.current_frame >= (len(traj) - 1) * self.frames_per_transition)
            if not reached:
                gi, gj = goal
                center_x = (gj + 0.5) * self.cell_size
                center_y = (gi + 0.5) * self.cell_size
                radius = self.cell_size * 0.3
                outline_color = self.agent_colors[idx % len(self.agent_colors)]
                fill_color = self.goal_colors[idx % len(self.agent_colors)]
                self.canvas.create_oval(center_x - radius, center_y - radius,
                                        center_x + radius, center_y + radius,
                                        fill=fill_color, outline=outline_color, width=2)
                self.canvas.create_text(center_x, center_y, text=f'{idx+1}',
                                        fill=outline_color, font=("Helvetica", 12, "bold"))
    
    def draw_agents(self):
        # Draw all agents - those at goal stay visible indefinitely
        for idx, traj in enumerate(self.trajectories):
            if not traj:
                continue
            seg_index = self.current_frame // self.frames_per_transition
            frac = (self.current_frame % self.frames_per_transition) / self.frames_per_transition

            # Determine current position
            if seg_index >= len(traj) - 1:
                pos = np.array(traj[-1], dtype=float)
                reached = True
            else:
                p0 = np.array(traj[seg_index], dtype=float)
                p1 = np.array(traj[seg_index + 1], dtype=float)
                pos = p0 + frac * (p1 - p0)
                reached = False

            a_i, a_j = pos
            center_x = (a_j + 0.5) * self.cell_size
            center_y = (a_i + 0.5) * self.cell_size
            agent_color = self.agent_colors[idx % len(self.agent_colors)]
            label_color = self.goal_colors[idx % len(self.agent_colors)]

            # Draw agent circle (always visible, even after reaching goal)
            radius = self.cell_size * 0.4
            self.canvas.create_oval(center_x - radius, center_y - radius,
                                    center_x + radius, center_y + radius,
                                    fill=agent_color, outline="")
            self.canvas.create_text(center_x, center_y, text=f'{idx+1}',
                                    fill=label_color, font=("Helvetica", 12, "bold"))
