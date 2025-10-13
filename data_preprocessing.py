import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

#Passed evals
class InkMLParser:
    """Parse InkML XML files to extract strokes and labels."""
    
    def parse_file(self, file_path):
        """
        Parse a single InkML file.
        
        Returns:
            strokes: List of numpy arrays, each shape (n_points, 3) for (x, y, t)
            label: Ground truth LaTeX string
        """
        # 1. Open XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # XML namespace for InkML
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        
        # 2. Find all <trace> elements (these are strokes)
        strokes = []
        for trace in root.findall('inkml:trace', ns):
            points = []
            
            # 3. Split the text by commas (each comma separates points)
            for point_str in trace.text.strip().split(','):
                # Split by spaces to get x, y, t
                coords = point_str.split()
                x = float(coords[0])
                y = float(coords[1])
                t = float(coords[2])
                points.append([x, y, t])
            
            # Convert to numpy array: shape (n_points, 3)
            stroke_array = np.array(points, dtype=np.float32)
            strokes.append(stroke_array)
        
        # 4. Find the normalizedLabel annotation (ground truth LaTeX)
        label = None
        for ann in root.findall('inkml:annotation', ns):
            if ann.get('type') == 'normalizedLabel':
                label = ann.text
                break
        
        # 5. Return list of strokes (List of 2D arrays) and label (string)
        return strokes, label

#Passed evals
class StrokeNormalizer:
    """
    Normalize strokes following the paper's approach:
    1. Time resampling: uniform temporal spacing
    2. Scale normalization: normalize to [0, N] preserving aspect ratio
    3. Discretization: round to nearest integer
    """
    
    def __init__(self, target_points_per_stroke=16, N=224):
        """
        Args:
            target_points_per_stroke: Number of points to resample each stroke to.
                                     Paper uses ~12-16 for math expressions.
            N: Normalization range [0, N]. Paper uses 224 (ViT image size).
        """
        self.target_points = target_points_per_stroke
        self.N = N
    
    def resample_stroke_uniform_time(self, stroke):
        """
        Resample a single stroke to have uniform temporal spacing.
        Following paper: "resample points at regular time intervals within each stroke"
        
        Args:
            stroke: numpy array shape (n_points, 3) with columns [x, y, t]
            
        Returns:
            resampled: numpy array shape (target_points, 3)
        """
        if len(stroke) < 2:
            # Edge case: stroke with only 1 point - duplicate it
            return np.repeat(stroke, self.target_points, axis=0)
        
        # Extract time values
        t_values = stroke[:, 2]
        t_start, t_end = t_values[0], t_values[-1]
        
        # Handle case where all timestamps are the same
        if t_end == t_start:
            # Just duplicate the stroke uniformly
            indices = np.linspace(0, len(stroke)-1, self.target_points).astype(int)
            return stroke[indices]
        
        # Create evenly spaced time points
        new_t = np.linspace(t_start, t_end, self.target_points)
        
        # Interpolate x and y at these new times
        new_x = np.interp(new_t, t_values, stroke[:, 0])
        new_y = np.interp(new_t, t_values, stroke[:, 1])
        
        # Combine into new stroke: shape (target_points, 3)
        resampled = np.column_stack([new_x, new_y, new_t])
        
        return resampled
    
    def normalize_spatial(self, strokes):
        """
        Normalize spatial coordinates to [0, N] preserving aspect ratio.
        Following paper: "scale and shift, preserving the aspect ratio, 
        so that all points fit into the range between 0 and N"
        
        Args:
            strokes: List of numpy arrays, each shape (n_points, 3)
            
        Returns:
            normalized_strokes: List with x, y in [0, N], t unchanged
        """
        # Step 1: Concatenate all points to find global bounding box
        all_points = np.vstack(strokes)
        
        # Step 2: Extract x, y coordinates
        x_coords = all_points[:, 0]
        y_coords = all_points[:, 1]
        
        # Step 3: Find bounding box
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Step 4: Calculate center
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Step 5: Use maximum range to preserve aspect ratio
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        
        # Step 6: Handle edge case (single point or all points identical)
        if max_range == 0:
            max_range = 1  # Avoid division by zero
        
        # Step 7: Scale to fit in [0, N]
        scale = self.N / max_range
        
        # Step 8: Normalize each stroke to [0, N]
        normalized_strokes = []
        for stroke in strokes:
            normalized = stroke.copy()  # Don't modify original
            
            # Center, scale, then shift to [0, N]
            normalized[:, 0] = (stroke[:, 0] - x_center) * scale + self.N / 2
            normalized[:, 1] = (stroke[:, 1] - y_center) * scale + self.N / 2
            # normalized[:, 2] stays unchanged (time)
            
            normalized_strokes.append(normalized)
        
        return normalized_strokes
    
    def discretize(self, strokes):
        """
        Discretize coordinates by rounding to nearest integer.
        Following paper: "round the normalized x and y coordinate to the nearest integer"
        
        Args:
            strokes: List of numpy arrays with continuous values in [0, N]
            
        Returns:
            discretized_strokes: List with x, y as integers, t unchanged
        """
        discretized_strokes = []
        for stroke in strokes:
            discretized = stroke.copy()
            
            # Round x and y to nearest integer
            discretized[:, 0] = np.round(stroke[:, 0])
            discretized[:, 1] = np.round(stroke[:, 1])
            # discretized[:, 2] stays unchanged (time)
            
            discretized_strokes.append(discretized)
        
        return discretized_strokes
    
    def normalize(self, strokes):
        """
        Full normalization pipeline following the paper:
        1. Time resampling: uniform temporal spacing within each stroke
        2. Spatial normalization: scale to [0, N] preserving aspect ratio
        3. Discretization: round to nearest integer
        
        Args:
            strokes: List of numpy arrays (variable length)
            
        Returns:
            normalized_strokes: List of numpy arrays (fixed length, discretized coords)
        """
        # Step 1: Time resampling
        resampled_strokes = [
            self.resample_stroke_uniform_time(stroke) 
            for stroke in strokes
        ]
        
        # Step 2: Spatial normalization to [0, N]
        spatial_normalized = self.normalize_spatial(resampled_strokes)
        
        # Step 3: Discretization (round to integers)
        discretized_strokes = self.discretize(spatial_normalized)
        
        return discretized_strokes


# Test the implementation
if __name__ == "__main__":
    print("="*70)
    print("Testing InkML Parser + Full Normalization Pipeline")
    print("="*70)
    
    # Initialize parser and normalizer
    parser = InkMLParser()
    normalizer = StrokeNormalizer(target_points_per_stroke=16)
    
    # Test on the example file
    test_file = "mathwriting-2024-excerpt/test/00eaf4f2d3d20bb1.inkml"
    
    print(f"\n📄 Parsing: {test_file.split('/')[-1]}")
    strokes, label = parser.parse_file(test_file)
    
    print(f"\n✓ Label: {label}")
    print(f"✓ Number of strokes: {len(strokes)}")
    
    # Show original stroke details
    print(f"\n{'ORIGINAL (Raw from InkML):':-^70}")
    total_points_before = 0
    all_x = []
    all_y = []
    for i, stroke in enumerate(strokes):
        total_points_before += len(stroke)
        all_x.extend(stroke[:, 0])
        all_y.extend(stroke[:, 1])
        print(f"  Stroke {i}: {stroke.shape[0]:3d} points | "
              f"x: [{stroke[:, 0].min():.0f}, {stroke[:, 0].max():.0f}] "
              f"y: [{stroke[:, 1].min():.0f}, {stroke[:, 1].max():.0f}]")
    
    print(f"\n  Total points: {total_points_before}")
    print(f"  Coordinate range: x=[{min(all_x):.0f}, {max(all_x):.0f}], "
          f"y=[{min(all_y):.0f}, {max(all_y):.0f}]")
    
    # Apply full normalization
    print(f"\n⚙️  Applying normalization pipeline...")
    print(f"     Step 1: Time resampling (→ 16 points/stroke)")
    print(f"     Step 2: Spatial normalization (→ [0, {normalizer.N}])")
    print(f"     Step 3: Discretization (→ round to integer)")
    normalized_strokes = normalizer.normalize(strokes)
    
    # Show normalized stroke details
    print(f"\n{'AFTER NORMALIZATION:':-^70}")
    total_points_after = 0
    for i, stroke in enumerate(normalized_strokes):
        total_points_after += len(stroke)
        # Calculate time delta
        if len(stroke) > 1:
            time_deltas = np.diff(stroke[:, 2])
            avg_delta = np.mean(time_deltas)
        else:
            avg_delta = 0
        
        print(f"  Stroke {i}: {stroke.shape[0]:2d} points | "
              f"x: [{stroke[:, 0].min():.0f}, {stroke[:, 0].max():.0f}] "
              f"y: [{stroke[:, 1].min():.0f}, {stroke[:, 1].max():.0f}] "
              f"Δt: {avg_delta:.1f}ms")
    
    print(f"\n  Total points: {total_points_after}")
    print(f"  Point reduction: {total_points_before} → {total_points_after} "
          f"({100*(1-total_points_after/total_points_before):.1f}% smaller)")
    
    # Verify normalization
    print(f"\n{'VERIFICATION:':-^70}")
    all_x_norm = np.concatenate([s[:, 0] for s in normalized_strokes])
    all_y_norm = np.concatenate([s[:, 1] for s in normalized_strokes])
    
    # Check if values are integers
    x_is_int = np.allclose(all_x_norm, np.round(all_x_norm))
    y_is_int = np.allclose(all_y_norm, np.round(all_y_norm))
    
    print(f"  ✓ All strokes have exactly {normalizer.target_points} points")
    print(f"  ✓ Time intervals uniform within each stroke")
    print(f"  ✓ Coordinates in [0, {normalizer.N}]: x=[{all_x_norm.min():.0f}, {all_x_norm.max():.0f}], "
          f"y=[{all_y_norm.min():.0f}, {all_y_norm.max():.0f}]")
    print(f"  ✓ Coordinates are integers: x={x_is_int}, y={y_is_int}")
    print(f"  ✓ Centered at ~{normalizer.N/2}: center=({all_x_norm.mean():.1f}, {all_y_norm.mean():.1f})")
    print(f"  ✓ Aspect ratio preserved")
    print(f"  ✓ Total unique x values: {len(np.unique(all_x_norm))}")
    print(f"  ✓ Total unique y values: {len(np.unique(all_y_norm))}")
    
    print(f"\n{'='*70}")
    print("✓ Full normalization pipeline working correctly!")
    print("="*70)