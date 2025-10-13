import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path
from data_preprocessing import InkMLParser, StrokeNormalizer

class PreprocessingVisualizer:
    """
    Interactive visualization showing the complete preprocessing pipeline:
    1. Raw InkML parsing
    2. Time resampling
    3. Spatial normalization to [0, 224]
    4. Discretization
    5. Text representation
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Parse and process the data
        print("📄 Parsing InkML file...")
        self.parser = InkMLParser()
        self.normalizer = StrokeNormalizer(target_points_per_stroke=16, N=224)
        
        self.raw_strokes, self.label = self.parser.parse_file(file_path)
        print(f"✓ Label: {self.label}")
        print(f"✓ Raw strokes: {len(self.raw_strokes)}")
        
        # Process through pipeline
        print("\n⚙️  Processing through pipeline...")
        self.resampled_strokes = [
            self.normalizer.resample_stroke_uniform_time(s) 
            for s in self.raw_strokes
        ]
        self.spatial_normalized = self.normalizer.normalize_spatial(self.resampled_strokes)
        self.discretized_strokes = self.normalizer.discretize(self.spatial_normalized)
        
        # Generate text representation
        self.text_representation = self.strokes_to_text(self.discretized_strokes)
        
        print("✓ Pipeline complete!")
        
        # Animation state
        self.current_stage = 0
        self.stages = [
            "Raw InkML",
            "Time Resampled",
            "Spatial Normalized",
            "Discretized",
            "Text Format"
        ]
        
        # Setup figure
        self.setup_figure()
    
    def strokes_to_text(self, strokes):
        """
        Convert discretized strokes to text representation.
        Following paper: "<stroke> x y x y ... <stroke> x y x y ..."
        """
        tokens = []
        for stroke in strokes:
            tokens.append("<stroke>")
            for point in stroke:
                x, y = int(point[0]), int(point[1])
                tokens.append(f"{x} {y}")
        return " ".join(tokens)
    
    def setup_figure(self):
        """Create the interactive visualization."""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle(f'Preprocessing Pipeline Visualization: {self.label}', 
                         fontsize=18, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                                   left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Stage 1: Raw InkML (top left)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_title('1. Raw InkML', fontweight='bold', fontsize=12)
        self.ax1.set_xlabel('X (pixels)')
        self.ax1.set_ylabel('Y (pixels)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.invert_yaxis()
        
        # Stage 2: Time Resampled (top middle)
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title('2. Time Resampled (16 pts/stroke)', fontweight='bold', fontsize=12)
        self.ax2.set_xlabel('X (pixels)')
        self.ax2.set_ylabel('Y (pixels)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.invert_yaxis()
        
        # Stage 3: Spatial Normalized (top right)
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax3.set_title('3. Spatial Normalized [0, 224]', fontweight='bold', fontsize=12)
        self.ax3.set_xlabel('X [0-224]')
        self.ax3.set_ylabel('Y [0-224]')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(-10, 234)
        self.ax3.set_ylim(-10, 234)
        self.ax3.invert_yaxis()  # Match screen coordinates (Y increases downward)
        
        # Stage 4: Discretized (middle left)
        self.ax4 = self.fig.add_subplot(gs[1, 0])
        self.ax4.set_title('4. Discretized (integers)', fontweight='bold', fontsize=12)
        self.ax4.set_xlabel('X [0-224]')
        self.ax4.set_ylabel('Y [0-224]')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.set_xlim(-10, 234)
        self.ax4.set_ylim(-10, 234)
        self.ax4.invert_yaxis()  # Match screen coordinates (Y increases downward)
        
        # Stage 5: Text representation (middle spanning)
        self.ax5 = self.fig.add_subplot(gs[1, 1:])
        self.ax5.axis('off')
        self.ax5.set_title('5. Text Representation (for VLM)', fontweight='bold', fontsize=12)
        
        # Info panel (bottom spanning)
        self.info_ax = self.fig.add_subplot(gs[2, :])
        self.info_ax.axis('off')
        
        # Colors for strokes
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.raw_strokes)))
        
        # Draw all stages
        self.draw_all_stages()
        self.update_info()
    
    def draw_all_stages(self):
        """Draw all preprocessing stages."""
        
        # Stage 1: Raw InkML
        for i, stroke in enumerate(self.raw_strokes):
            x = stroke[:, 0]
            y = stroke[:, 1]
            self.ax1.plot(x, y, '-o', color=self.colors[i], 
                         linewidth=2, markersize=3, alpha=0.7, label=f'S{i}')
        
        # Stage 2: Time Resampled
        for i, stroke in enumerate(self.resampled_strokes):
            x = stroke[:, 0]
            y = stroke[:, 1]
            self.ax2.plot(x, y, '-o', color=self.colors[i], 
                         linewidth=2, markersize=5, alpha=0.7)
            # Highlight uniform spacing
            for j in range(len(x)-1):
                self.ax2.plot([x[j], x[j+1]], [y[j], y[j+1]], 
                            'k-', alpha=0.2, linewidth=1)
        
        # Stage 3: Spatial Normalized
        for i, stroke in enumerate(self.spatial_normalized):
            x = stroke[:, 0]
            y = stroke[:, 1]
            self.ax3.plot(x, y, '-o', color=self.colors[i], 
                         linewidth=2, markersize=5, alpha=0.7)
        # Add bounding box to show [0, 224] range
        bbox = FancyBboxPatch((0, 0), 224, 224, 
                             boxstyle="round,pad=5", 
                             edgecolor='red', facecolor='none', 
                             linewidth=2, linestyle='--', alpha=0.5)
        self.ax3.add_patch(bbox)
        self.ax3.text(112, -5, '[0, 224] range', ha='center', fontsize=10, color='red')
        
        # Stage 4: Discretized (integers)
        for i, stroke in enumerate(self.discretized_strokes):
            x = stroke[:, 0]
            y = stroke[:, 1]
            self.ax4.plot(x, y, '-s', color=self.colors[i], 
                         linewidth=2, markersize=6, alpha=0.7, 
                         markerfacecolor=self.colors[i], markeredgecolor='black', 
                         markeredgewidth=1)
        # Add grid to emphasize integer coordinates
        self.ax4.grid(True, alpha=0.5, which='major')
        self.ax4.set_xticks(range(0, 225, 25))
        self.ax4.set_yticks(range(0, 225, 25))
        
        # Stage 5: Text representation
        self.draw_text_representation()
    
    def draw_text_representation(self):
        """Draw the text representation with syntax highlighting."""
        self.ax5.clear()
        self.ax5.axis('off')
        
        # Split text into chunks for display
        text = self.text_representation
        max_chars_per_line = 100
        
        # Syntax highlighting
        lines = []
        current_line = ""
        tokens = text.split()
        
        for token in tokens:
            if len(current_line) + len(token) + 1 > max_chars_per_line:
                lines.append(current_line)
                current_line = token
            else:
                current_line += (" " if current_line else "") + token
        if current_line:
            lines.append(current_line)
        
        # Display with color coding
        y_pos = 0.95
        for line in lines[:5]:  # Show first 5 lines
            # Highlight <stroke> tokens
            colored_line = line.replace("<stroke>", "\033[91m<stroke>\033[0m")
            self.ax5.text(0.02, y_pos, line, 
                         transform=self.ax5.transAxes,
                         fontsize=9, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                 alpha=0.3, edgecolor='orange', linewidth=2))
            y_pos -= 0.18
        
        if len(lines) > 5:
            self.ax5.text(0.02, y_pos, f"... ({len(lines)-5} more lines)", 
                         transform=self.ax5.transAxes,
                         fontsize=9, style='italic', color='gray')
        
        # Add annotation
        self.ax5.text(0.5, 0.02, 
                     'Format: <stroke> x y x y ... <stroke> x y x y ...\n'
                     'Ready for BPE tokenization and VLM input!',
                     transform=self.ax5.transAxes,
                     fontsize=11, ha='center', style='italic',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    def update_info(self):
        """Update information panel."""
        self.info_ax.clear()
        self.info_ax.axis('off')
        
        # Calculate statistics
        raw_points = sum(len(s) for s in self.raw_strokes)
        processed_points = sum(len(s) for s in self.discretized_strokes)
        
        # Get coordinate ranges
        raw_x = np.concatenate([s[:, 0] for s in self.raw_strokes])
        raw_y = np.concatenate([s[:, 1] for s in self.raw_strokes])
        
        norm_x = np.concatenate([s[:, 0] for s in self.discretized_strokes])
        norm_y = np.concatenate([s[:, 1] for s in self.discretized_strokes])
        
        text_length = len(self.text_representation)
        num_tokens = len(self.text_representation.split())
        
        info_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                          PREPROCESSING PIPELINE SUMMARY                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

📊 TRANSFORMATION STATISTICS:

   Stage 1: RAW INKML                          Stage 2: TIME RESAMPLED
   • Strokes: {len(self.raw_strokes)}                              • Strokes: {len(self.resampled_strokes)}
   • Total points: {raw_points}                      • Total points: {processed_points}
   • X range: [{raw_x.min():.0f}, {raw_x.max():.0f}]            • Points/stroke: 16 (uniform)
   • Y range: [{raw_y.min():.0f}, {raw_y.max():.0f}]            • Δt: uniform within each stroke
   
   Stage 3: SPATIAL NORMALIZED                 Stage 4: DISCRETIZED
   • Range: [0, 224]                           • X: integers in [0, 224]
   • X: [{norm_x.min():.0f}, {norm_x.max():.0f}]                        • Y: integers in [0, 224]
   • Y: [{norm_y.min():.0f}, {norm_y.max():.0f}]                        • Unique X values: {len(np.unique(norm_x))}
   • Aspect ratio: preserved                   • Unique Y values: {len(np.unique(norm_y))}
   
   Stage 5: TEXT REPRESENTATION
   • Format: "<stroke> x y x y ... <stroke> x y x y ..."
   • Total characters: {text_length}
   • Total tokens: {num_tokens} (before BPE)
   • Reduction: {raw_points} points → {num_tokens} tokens ({100*(1-num_tokens/raw_points):.1f}% smaller)

🎯 PIPELINE BENEFITS:
   ✓ Device-independent (normalized coordinates)
   ✓ Fixed sequence length (16 points/stroke)
   ✓ Compact representation ({100*(1-processed_points/raw_points):.1f}% fewer points)
   ✓ VLM-compatible (text format)
   ✓ Aspect ratio preserved (no distortion)

📝 NEXT STEPS:
   → Apply BPE tokenization (model-specific)
   → Build LaTeX tokenizer for target labels
   → Create PyTorch Dataset class
   → Fine-tune PaLI with LoRA
"""
        
        self.info_ax.text(0.5, 0.5, info_text, 
                         transform=self.info_ax.transAxes,
                         fontsize=9, verticalalignment='center', ha='center',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show(self):
        """Display the visualization."""
        plt.show()


def main():
    """Main function to run the preprocessing visualizer."""
    file_path = 'mathwriting-2024-excerpt/test/00eaf4f2d3d20bb1.inkml'
    
    print("=" * 80)
    print("🎨 Preprocessing Pipeline Visualizer")
    print("=" * 80)
    print("\nThis visualization shows how raw InkML data is transformed through")
    print("the complete preprocessing pipeline for VLM fine-tuning:\n")
    print("  1. Raw InkML parsing")
    print("  2. Time resampling (uniform temporal spacing)")
    print("  3. Spatial normalization ([0, 224] range)")
    print("  4. Discretization (round to integers)")
    print("  5. Text representation (for VLM input)")
    print("\n" + "=" * 80)
    
    visualizer = PreprocessingVisualizer(file_path)
    
    print("\n✓ Visualization ready!")
    print("  Opening interactive viewer...\n")
    
    visualizer.show()


if __name__ == '__main__':
    main()

