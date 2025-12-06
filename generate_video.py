#!/usr/bin/env python3
"""
MAPF Video Generator
Generate MP4 videos from MAPF JSON log files.

Usage:
    python generate_video.py <log_file> [options]

Examples:
    python generate_video.py logs/experiment.json
    python generate_video.py logs/experiment.json --fps 30 --cell-size 40
"""

import json
import argparse
import sys
import os
import tempfile
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def hex_to_rgb(hex_color):
    """Convert a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def blend_with_white(hex_color, mix=0.7):
    """Return a lighter shade of hex_color by blending with white."""
    rgb = hex_to_rgb(hex_color)
    blended_rgb = tuple(round((1 - mix) * c + mix * 255) for c in rgb)
    return blended_rgb


def darken_color(hex_color, mix=0.3):
    """Return a darker version of hex_color by blending with black."""
    rgb = hex_to_rgb(hex_color)
    dark_rgb = tuple(round((1 - mix) * c) for c in rgb)
    return dark_rgb


class VideoGenerator:
    def __init__(self, log_data, cell_size=50, fps=20, frames_per_step=10):
        self.log_data = log_data
        self.cell_size = cell_size
        self.fps = fps
        self.frames_per_step = frames_per_step

        # Extract grid size
        grid_size = log_data['environment']['gridSize']
        self.rows, self.cols = grid_size

        # Create grid with obstacles
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        for obs in log_data['environment']['obstacles']:
            r, c = obs['cell']
            self.grid[r, c] = -1

        # Extract agent data
        self.agent_starts = []
        self.agent_goals = []
        for agent in log_data['agents']:
            start = tuple(agent['initialState']['cell'])
            goal = tuple(agent['goalState']['cell'])
            self.agent_starts.append(start)
            self.agent_goals.append(goal)

        # Extract trajectories
        self.trajectories = self._extract_trajectories()

        # Colors
        self.agent_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b',
                            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
        self.goal_colors = [blend_with_white(c, mix=0.7) for c in self.agent_colors]

        # Canvas dimensions (ensure even numbers for video encoding)
        self.canvas_width = self.cols * self.cell_size
        self.canvas_height = self.rows * self.cell_size
        # Make dimensions even for libx264
        if self.canvas_width % 2 != 0:
            self.canvas_width += 1
        if self.canvas_height % 2 != 0:
            self.canvas_height += 1

        # Calculate max timesteps
        self.max_timesteps = max(len(traj) for traj in self.trajectories if traj)

    def _extract_trajectories(self):
        """Extract final trajectories from the jointPlan."""
        final_subplan_ids = self.log_data['jointPlan']['subplans']
        final_trajectories = []

        for subplan_id in final_subplan_ids:
            found = False
            for subplan in self.log_data.get('agentSubplans', []):
                if subplan['id'] == subplan_id:
                    traj = [tuple(step['cell']) for step in subplan['steps']]
                    final_trajectories.append(traj)
                    found = True
                    break

            if not found:
                for path in self.log_data.get('agentPaths', []):
                    if path['subplanId'] == subplan_id:
                        traj = [tuple(step['cell']) for step in path['steps']]
                        final_trajectories.append(traj)
                        break

        return final_trajectories

    def _get_agent_position(self, agent_idx, frame):
        """Get interpolated agent position at given frame."""
        traj = self.trajectories[agent_idx]
        if not traj:
            return None

        step_idx = frame // self.frames_per_step
        frac = (frame % self.frames_per_step) / self.frames_per_step

        if step_idx >= len(traj) - 1:
            return np.array(traj[-1], dtype=float)
        else:
            p0 = np.array(traj[step_idx], dtype=float)
            p1 = np.array(traj[step_idx + 1], dtype=float)
            return p0 + frac * (p1 - p0)

    def _draw_frame(self, frame_num):
        """Render a single frame as PIL Image."""
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        for i in range(self.rows + 1):
            y = i * self.cell_size
            draw.line([(0, y), (self.canvas_width, y)], fill='lightgray', width=1)
        for j in range(self.cols + 1):
            x = j * self.cell_size
            draw.line([(x, 0), (x, self.canvas_height)], fill='lightgray', width=1)

        # Draw obstacles
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == -1:
                    x1, y1 = j * self.cell_size, i * self.cell_size
                    x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                    draw.rectangle([x1, y1, x2, y2], fill='#444444', outline='#333333')

        # Draw goal circles (for agents that haven't reached goal yet)
        for idx, goal in enumerate(self.agent_goals):
            traj = self.trajectories[idx]
            total_frames_for_agent = (len(traj) - 1) * self.frames_per_step
            reached = frame_num >= total_frames_for_agent

            if not reached:
                gi, gj = goal
                center_x = (gj + 0.5) * self.cell_size
                center_y = (gi + 0.5) * self.cell_size
                radius = self.cell_size * 0.3
                color_idx = idx % len(self.agent_colors)
                fill_color = self.goal_colors[color_idx]
                outline_color = hex_to_rgb(self.agent_colors[color_idx])

                draw.ellipse(
                    [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    fill=fill_color, outline=outline_color, width=2
                )

        # Draw agents
        for idx in range(len(self.trajectories)):
            pos = self._get_agent_position(idx, frame_num)
            if pos is None:
                continue

            a_i, a_j = pos
            center_x = (a_j + 0.5) * self.cell_size
            center_y = (a_i + 0.5) * self.cell_size
            radius = self.cell_size * 0.4

            color_idx = idx % len(self.agent_colors)
            agent_color = hex_to_rgb(self.agent_colors[color_idx])

            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=agent_color
            )

            # Draw agent number
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(self.cell_size * 0.4))
            except:
                font = ImageFont.load_default()

            text = str(idx + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = center_x - text_width / 2
            text_y = center_y - text_height / 2 - 2

            label_color = self.goal_colors[color_idx]
            draw.text((text_x, text_y), text, fill=label_color, font=font)

        # Draw timestep info
        current_step = frame_num // self.frames_per_step + 1
        info_text = f"Step: {current_step}/{self.max_timesteps}"
        try:
            info_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            info_font = ImageFont.load_default()
        draw.rectangle([5, 5, 120, 25], fill='white', outline='gray')
        draw.text((10, 7), info_text, fill='black', font=info_font)

        return img

    def generate_video(self, output_path):
        """Generate video from all frames."""
        total_frames = (self.max_timesteps - 1) * self.frames_per_step + 1

        print(f"Generating video with {total_frames} frames...")
        print(f"  Grid: {self.rows}x{self.cols}")
        print(f"  Agents: {len(self.trajectories)}")
        print(f"  Timesteps: {self.max_timesteps}")
        print(f"  FPS: {self.fps}")

        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()

        try:
            # Generate all frames
            for frame_num in range(total_frames):
                if frame_num % 50 == 0:
                    print(f"  Rendering frame {frame_num + 1}/{total_frames}...")

                img = self._draw_frame(frame_num)
                frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.png")
                img.save(frame_path)

            print(f"  Encoding video with ffmpeg...")

            # Use ffmpeg to create video
            # Try libopenh264 first (available in conda ffmpeg), fallback to libx264
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libopenh264',
                '-pix_fmt', 'yuv420p',
                '-b:v', '2M',
                output_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ffmpeg stderr: {result.stderr}")
                raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
            print(f"  Video saved to: {output_path}")

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir)


def generate_video_from_log(log_file, cell_size=50, fps=20, frames_per_step=10, output_path=None):
    """Main function to generate video from a log file."""

    # Load log file
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(log_file)[0]
        output_path = f"{base_name}.mp4"

    # Create video generator
    generator = VideoGenerator(
        log_data,
        cell_size=cell_size,
        fps=fps,
        frames_per_step=frames_per_step
    )

    # Generate video
    generator.generate_video(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate MP4 video from MAPF JSON log files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s logs/experiment.json
  %(prog)s logs/experiment.json --fps 30
  %(prog)s logs/experiment.json --cell-size 60 --frames-per-step 5
  %(prog)s logs/experiment.json -o custom_output.mp4
        """
    )

    parser.add_argument(
        'log_file',
        help='Path to the JSON log file'
    )

    parser.add_argument(
        '--cell-size',
        type=int,
        default=50,
        metavar='SIZE',
        help='Size of each grid cell in pixels (default: 50)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        metavar='FPS',
        help='Video frames per second (default: 20)'
    )

    parser.add_argument(
        '--frames-per-step',
        type=int,
        default=10,
        metavar='N',
        help='Animation frames per timestep for smooth motion (default: 10)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        metavar='PATH',
        help='Output video path (default: same as log file with .mp4 extension)'
    )

    args = parser.parse_args()

    generate_video_from_log(
        args.log_file,
        cell_size=args.cell_size,
        fps=args.fps,
        frames_per_step=args.frames_per_step,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
