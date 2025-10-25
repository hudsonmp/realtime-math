import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib.patches as mpatches
from pathlib import Path

class InteractiveInkMLViewer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.label, self.traces = self.parse_inkml(file_path)
        self.current_frame = 0
        self.is_playing = False
        self.speed = 1.0
        self.show_strokes = [True] * len(self.traces)
        
        # Setup the figure
        self.setup_figure()
        
    def parse_inkml(self, file_path):
        """Parse an InkML file and extract stroke data."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        # Extract label
        label = root.find('.//ink:annotation[@type="label"]', ns)
        label_text = label.text if label is not None else "No label"
        
        # Extract all traces (strokes)
        traces = []
        for trace in root.findall('.//ink:trace', ns):
            points = []
            trace_data = trace.text.strip().split(',')
            for point in trace_data:
                coords = point.strip().split()
                if len(coords) == 3:
                    x, y, t = float(coords[0]), float(coords[1]), float(coords[2])
                    points.append((x, y, t))
            traces.append(points)
        
        return label_text, traces
    
    def setup_figure(self):
        """Create the interactive figure with controls."""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(f'Interactive InkML Viewer: {self.label}', 
                         fontsize=16, fontweight='bold')
        
        # Create main plot
        self.ax = plt.subplot(1, 2, 1)
        self.title_text = self.ax.set_title('Animated Stroke Drawing - Time: 0.0ms', 
                                           fontweight='bold', fontsize=14)
        self.ax.set_xlabel('X coordinate (pixels)')
        self.ax.set_ylabel('Y coordinate (pixels)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Create info panel
        self.info_ax = plt.subplot(1, 2, 2)
        self.info_ax.axis('off')
        
        # Color palette for strokes
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.traces)))
        
        # Initialize line objects
        self.lines = []
        self.points = []
        for i in range(len(self.traces)):
            line, = self.ax.plot([], [], '-', color=self.colors[i], 
                                linewidth=3, alpha=0.8, label=f'Stroke {i}')
            point, = self.ax.plot([], [], 'o', color=self.colors[i], 
                                 markersize=8, alpha=0.9)
            self.lines.append(line)
            self.points.append(point)
        
        # Set axis limits
        all_x = [p[0] for trace in self.traces for p in trace]
        all_y = [p[1] for trace in self.traces for p in trace]
        margin = 50
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # CRITICAL: Invert Y-axis AFTER setting limits!
        # InkML uses screen coordinates where Y increases downward (0 at top)
        # but matplotlib default has Y increasing upward (0 at bottom)
        self.ax.invert_yaxis()
        
        # Create controls
        self.create_controls()
        
        # Display info
        self.update_info()
        
        # Setup animation
        self.anim = FuncAnimation(self.fig, self.animate, 
                                 frames=self.get_total_frames(),
                                 interval=20, blit=False, repeat=True)
        # Start playing automatically
        self.is_playing = True
        self.play_button.label.set_text('Pause')
        self.play_button.color = 'lightcoral'
        
    def create_controls(self):
        """Create interactive controls."""
        # Play/Pause button
        self.play_ax = plt.axes([0.15, 0.02, 0.08, 0.04])
        self.play_button = Button(self.play_ax, 'Play', color='lightgreen')
        self.play_button.on_clicked(self.toggle_play)
        
        # Reset button
        self.reset_ax = plt.axes([0.25, 0.02, 0.08, 0.04])
        self.reset_button = Button(self.reset_ax, 'Reset', color='lightcoral')
        self.reset_button.on_clicked(self.reset)
        
        # Speed slider
        self.speed_ax = plt.axes([0.15, 0.08, 0.3, 0.02])
        self.speed_slider = Slider(self.speed_ax, 'Speed', 0.1, 5.0, 
                                   valinit=1.0, valstep=0.1)
        self.speed_slider.on_changed(self.update_speed)
        
    def get_total_frames(self):
        """Calculate total animation frames."""
        # Get total time range
        all_times = [p[2] for trace in self.traces for p in trace]
        if not all_times:
            return 100
        total_time = max(all_times) - min(all_times)
        # Convert to frames (more frames = smoother animation)
        return max(200, int(total_time / 10))  # ~10ms per frame
    
    def animate(self, frame):
        """Animation function - draws strokes in real-time based on timestamps."""
        # Calculate current time based on frame
        all_times = [p[2] for trace in self.traces for p in trace]
        if not all_times:
            return self.lines + self.points
        
        min_time = min(all_times)
        max_time = max(all_times)
        total_time = max_time - min_time
        
        # Current time in the animation (affected by speed)
        current_time = min_time + (frame / self.get_total_frames()) * total_time * self.speed
        
        # Update title with current time
        self.title_text.set_text(f'Animated Stroke Drawing - Time: {current_time:.1f}ms')
        
        # Update each stroke based on current time
        for i, trace in enumerate(self.traces):
            if not self.show_strokes[i] or not trace:
                self.lines[i].set_data([], [])
                self.points[i].set_data([], [])
                continue
            
            # Find all points that should be visible at current time
            visible_points = []
            for point in trace:
                if point[2] <= current_time:
                    visible_points.append(point)
                else:
                    break
            
            if visible_points:
                x_data = [p[0] for p in visible_points]
                y_data = [p[1] for p in visible_points]
                self.lines[i].set_data(x_data, y_data)
                
                # Show current drawing point (pulsing effect)
                if len(visible_points) < len(trace):
                    self.points[i].set_data([x_data[-1]], [y_data[-1]])
                else:
                    # Stroke is complete, no current point
                    self.points[i].set_data([], [])
            else:
                # Stroke hasn't started yet
                self.lines[i].set_data([], [])
                self.points[i].set_data([], [])
        
        return self.lines + self.points
    
    def toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.label.set_text('Pause')
            self.play_button.color = 'lightcoral'
            self.anim.event_source.start()
        else:
            self.play_button.label.set_text('Play')
            self.play_button.color = 'lightgreen'
            self.anim.event_source.stop()
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset animation."""
        self.is_playing = False
        self.play_button.label.set_text('Play')
        self.play_button.color = 'lightgreen'
        self.anim.event_source.stop()
        
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        
        self.anim.frame_seq = self.anim.new_frame_seq()
        self.fig.canvas.draw_idle()
    
    def update_speed(self, val):
        """Update animation speed."""
        self.speed = val
    
    def update_info(self):
        """Update information panel."""
        self.info_ax.clear()
        self.info_ax.axis('off')
        
        # Folder explanations
        folder_info = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         DATASET FOLDER STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 symbols/
   Individual math symbols (single characters)
   Examples: ∫, α, β, ∑, ∏, etc.
   Used for: Symbol recognition training
   
📁 train/
   Training data: Complete expressions
   Examples: E ↦ ∫_E f dμ
   Used for: Full expression recognition
   
📁 synthetic/
   Synthetically generated expressions
   Computer-generated handwriting variants
   Used for: Data augmentation
   
📁 test/
   Test set for model evaluation
   Held-out data for validation
   
📁 valid/
   Validation set for hyperparameter tuning
   Used during training to prevent overfitting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           INKML DATA FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each trace contains (X, Y, T) coordinates:
  • X: Horizontal position (pixels)
  • Y: Vertical position (pixels)  
  • T: Timestamp (milliseconds)

Strokes are drawn sequentially with timing
information preserved for realistic replay.
"""
        
        # File-specific info
        file_name = Path(self.file_path).stem
        folder = Path(self.file_path).parent.name
        
        # Statistics
        num_strokes = len(self.traces)
        total_points = sum(len(trace) for trace in self.traces)
        
        all_times = [p[2] for trace in self.traces for p in trace]
        total_time = max(all_times) - min(all_times) if all_times else 0
        
        file_info = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          CURRENT FILE INFO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Label: {self.label}
File: {file_name}
Folder: {folder}/

Number of Strokes: {num_strokes}
Total Points: {total_points}
Total Time: {total_time:.1f} ms
Average Points/Stroke: {total_points/num_strokes:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           STROKE DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, trace in enumerate(self.traces):
            duration = trace[-1][2] - trace[0][2] if len(trace) > 0 else 0
            file_info += f"\nStroke {i}: {len(trace)} pts, {duration:.0f}ms"
        
        # Combine and display
        full_text = folder_info + file_info
        
        self.info_ax.text(0.05, 0.98, full_text, 
                         transform=self.info_ax.transAxes,
                         fontsize=9, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add legend
        legend_elements = []
        for i in range(len(self.traces)):
            legend_elements.append(
                mpatches.Patch(color=self.colors[i], label=f'Stroke {i}')
            )
        self.info_ax.legend(handles=legend_elements, loc='lower right', 
                           fontsize=8, framealpha=0.9)
    
    def show(self):
        """Display the interactive viewer."""
        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
        plt.show()


def main():
    """Main function to run the interactive viewer."""
    file_path = '/Users/hudsonmitchell-pullman/realtime-math/mathwriting-2024-excerpt/train/03c3977784c8f56c.inkml'
    print("=" * 60)
    print("🎨 Interactive InkML Viewer")
    print("=" * 60)
    print("\nControls:")
    print("  • Play/Pause: Start/stop animation")
    print("  • Reset: Restart from beginning")
    print("  • Speed: Control playback speed (0.1x to 5x)")
    print("  • Stroke Checkboxes: Show/hide individual strokes")
    print("\nLoading file...")
    
    viewer = InteractiveInkMLViewer(file_path)
    
    print(f"✓ Loaded: {viewer.label}")
    print(f"✓ Strokes: {len(viewer.traces)}")
    print("\nOpening interactive viewer...")
    
    viewer.show()


if __name__ == '__main__':
    main()

