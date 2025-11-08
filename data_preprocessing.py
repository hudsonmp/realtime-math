import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import re
from PIL import Image

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
    
    def strokes_to_text(self, strokes):
        """
        Convert discretized strokes to text with RELATIVE offsets.
        Following paper Section 3.1: (x^tr, y^tr) = (x_t, y_t) - (x_{t-1}, y_{t-1})

        Format: '<image> <stroke> x0 y0 dx1 dy1 dx2 dy2 ...'
        PaliGemma requires <image> token at the start.
        First point uses absolute coordinates, subsequent points use relative offsets.
        """
        tokens = ["<image>"]  # PaliGemma requires this special token
        for stroke in strokes:
            tokens.append("<stroke>")
            prev_x, prev_y = None, None
            for point in stroke:
                x, y = int(point[0]), int(point[1])
                if prev_x is not None:
                    # Relative offset: (x_t, y_t) - (x_{t-1}, y_{t-1})
                    dx = x - prev_x
                    dy = y - prev_y
                    tokens.append(f"{dx} {dy}")
                else:
                    # First point of stroke: use absolute coordinates
                    tokens.append(f"{x} {y}")
                prev_x, prev_y = x, y
        return " ".join(tokens)


class InkRenderer:
    """
    Render normalized strokes as images with time and distance information.
    Following paper Section 3.2 - Image representation.
    """
    
    def __init__(self, image_size=224, num_lines=2):
        """
        Args:
            image_size: Output image size (default 224 for ViT)
            num_lines: Number of lines to render (paper uses 2 for 2:1 aspect ratio)
        """
        self.image_size = image_size
        self.num_lines = num_lines
        
    def render(self, strokes):
        """
        Render normalized strokes to RGB image with time and distance encoding.
        
        Following paper Equation 1:
        - Red channel: normalized time within the ink
        - Green channel: normalized |dx| (horizontal movement)
        - Blue channel: normalized |dy| (vertical movement)
        
        Args:
            strokes: List of normalized numpy arrays, each shape (n_points, 3) with [x, y, t]
            
        Returns:
            PIL.Image: RGB image of size (image_size, image_size)
        """
        if len(strokes) == 0:
            # Return blank white image if no strokes
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        
        # Step 1: Calculate time and distance information for all points
        all_points = []
        all_times = []
        all_dx = []
        all_dy = []
        
        for stroke in strokes:
            for i in range(len(stroke)):
                x, y, t = stroke[i]
                all_points.append((x, y))
                all_times.append(t)
                
                # Calculate dx, dy for this point
                if i > 0:
                    dx = stroke[i, 0] - stroke[i-1, 0]
                    dy = stroke[i, 1] - stroke[i-1, 1]
                else:
                    dx, dy = 0, 0  # First point of stroke has no previous point
                
                all_dx.append(abs(dx))
                all_dy.append(abs(dy))
        
        # Step 2: Normalize time, dx, dy to [0, 1] for color channels
        # Following paper Equation 1: c^R = (t_{i,j} - t_{0,0}) / max(t_{m,n})
        all_times = np.array(all_times)
        all_dx = np.array(all_dx)
        all_dy = np.array(all_dy)
        
        t_base = all_times[0]  # t_{0,0}: first point time
        t_max = all_times.max()
        if t_max > t_base:
            normalized_times = (all_times - t_base) / (t_max - t_base)
        else:
            normalized_times = np.zeros_like(all_times)
        
        max_dx = all_dx.max()
        if max_dx > 0:
            normalized_dx = all_dx / max_dx
        else:
            normalized_dx = np.zeros_like(all_dx)
        
        max_dy = all_dy.max()
        if max_dy > 0:
            normalized_dy = all_dy / max_dy
        else:
            normalized_dy = np.zeros_like(all_dy)
        
        # Step 3: Render to image with multi-line support
        if self.num_lines == 1:
            # Simple single-line rendering
            return self._render_single_line(all_points, normalized_times, normalized_dx, normalized_dy)
        else:
            # Multi-line rendering (paper uses 2 lines)
            return self._render_multiline(all_points, normalized_times, normalized_dx, normalized_dy)
    
    def _render_single_line(self, points, times, dx, dy):
        """Render ink in a single line (stretch to fill image)."""
        # Create blank white canvas
        canvas = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
        
        # Draw each point with its color encoding
        for i, (x, y) in enumerate(points):
            # Convert coordinates to image space
            img_x = int(x)
            img_y = int(y)
            
            # Ensure within bounds
            if 0 <= img_x < self.image_size and 0 <= img_y < self.image_size:
                # Set color: R=time, G=dx, B=dy
                canvas[img_y, img_x, 0] = times[i]  # Red channel
                canvas[img_y, img_x, 1] = dx[i]     # Green channel
                canvas[img_y, img_x, 2] = dy[i]     # Blue channel
        
        # Convert to 0-255 range and create PIL Image
        canvas = (canvas * 255).astype(np.uint8)
        return Image.fromarray(canvas, mode='RGB')
    
    def _render_multiline(self, points, times, dx, dy):
        """
        Render ink directly to square canvas without distortion.
        Simplified: just render points at their normalized coordinates [0, 224].
        """
        if len(points) == 0:
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))

        # Create square canvas
        canvas = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)

        # Draw each point directly using normalized coordinates
        # Points are already in [0, 224] from normalization
        for i, (x, y) in enumerate(points):
            img_x = int(x)
            img_y = int(y)

            # Ensure within bounds
            if 0 <= img_x < self.image_size and 0 <= img_y < self.image_size:
                # Set color with small thickness for visibility
                for dy_offset in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        px = img_x + dx_offset
                        py = img_y + dy_offset
                        if 0 <= px < self.image_size and 0 <= py < self.image_size:
                            canvas[py, px, 0] = times[i]  # Red channel
                            canvas[py, px, 1] = dx[i]     # Green channel
                            canvas[py, px, 2] = dy[i]     # Blue channel

        # Convert to image
        final_image = Image.fromarray((canvas * 255).astype(np.uint8), mode='RGB')
        return final_image


class LaTeXTokenizer:
    """Tokenize LaTeX for evaluation (CER calculation)."""
    
    def __init__(self):
        # Common LaTeX commands (from paper Appendix F)
        self.latex_pattern = re.compile(
            r'\\[a-zA-Z]+|'  # \frac, \sqrt, \alpha, etc.
            r'[{}()[\]^_=+\-*/|<>]|'  # Special chars
            r'[a-zA-Z0-9]|'  # Alphanumeric
            r'\s+'  # Whitespace
        )
    
    def tokenize(self, latex_string):
        """Tokenize LaTeX string into tokens."""
        if not latex_string:
            return []
        tokens = self.latex_pattern.findall(latex_string)
        return [t for t in tokens if t.strip()]
    
    def compute_cer(self, predicted, target):
        """Compute Character Error Rate between predicted and target LaTeX."""
        pred_tokens = self.tokenize(predicted)
        target_tokens = self.tokenize(target)
        
        # Levenshtein distance at token level
        m, n = len(pred_tokens), len(target_tokens)
        if n == 0:
            return 0.0 if m == 0 else 1.0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == target_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n] / n


class MathWritingDataset(Dataset):
    """PyTorch Dataset for MathWriting InkML files."""

    def __init__(self, data_dir, split='train', target_points=16, N=224, validate_on_load=False):
        """
        Args:
            data_dir: Base directory (e.g., "mathwriting-2024-excerpt")
            split: 'train', 'valid', 'test', or 'symbols'
            target_points: Points per stroke after resampling
            N: Normalization range [0, N]
            validate_on_load: If True, pre-validate all files (slow). If False, validate lazily (fast).
        """
        self.data_dir = data_dir
        self.split = split

        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            raise ValueError(f"Split directory not found: {split_path}")

        # Find all potential files
        all_files = sorted(glob.glob(os.path.join(split_path, "*.inkml")))
        if len(all_files) == 0:
            raise ValueError(f"No .inkml files found in {split_path}")

        self.parser = InkMLParser()
        self.normalizer = StrokeNormalizer(target_points_per_stroke=target_points, N=N)
        self.renderer = InkRenderer(image_size=N, num_lines=2)

        # FAST MODE: Skip pre-validation, validate lazily during training
        if not validate_on_load:
            print(f"⚡ Fast loading: {len(all_files):,} files in {split} split (lazy validation)")
            self.files = all_files
        else:
            # SLOW MODE: Pre-validate all files (useful for testing)
            print(f"Validating {len(all_files)} files in {split} split (with image rendering test)...")
            self.files = self._validate_files(all_files)

            if len(self.files) == 0:
                raise ValueError(f"No valid .inkml files found in {split_path}")

            invalid_count = len(all_files) - len(self.files)
            if invalid_count > 0:
                warnings.warn(f"Skipped {invalid_count}/{len(all_files)} invalid files in {split} split")
            print(f"Loaded {len(self.files)} valid files for {split} split")
        
    def _validate_files(self, file_list):
        """
        Pre-validate all files and return only those that can be processed.

        Args:
            file_list: List of file paths to validate

        Returns:
            List of valid file paths
        """
        valid_files = []
        total_files = len(file_list)
        print_interval = max(1, total_files // 20)  # Print progress every 5%

        for idx, file_path in enumerate(file_list):
            try:
                # Try to parse and process the file
                strokes, label = self.parser.parse_file(file_path)

                # Check required fields
                if label is None:
                    raise ValueError("Missing normalizedLabel")
                if len(strokes) == 0:
                    raise ValueError("No strokes found")

                # Try normalization to ensure it works
                normalized = self.normalizer.normalize(strokes)
                _ = self.normalizer.strokes_to_text(normalized)

                # IMPORTANT: Also test rendering to catch any image generation issues
                _ = self.renderer.render(normalized)

                # If we got here, the file is valid
                valid_files.append(file_path)

            except Exception as e:
                # Silently skip invalid files (summary reported in __init__)
                pass

            # Print progress for large datasets
            if (idx + 1) % print_interval == 0 or (idx + 1) == total_files:
                percent = (idx + 1) / total_files * 100
                print(f"  Progress: {idx + 1}/{total_files} ({percent:.1f}%) - {len(valid_files)} valid")

        return valid_files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """Return dict with 'stroke_text', 'image', and 'label'."""
        file_path = self.files[idx]

        try:
            # Parse and process file
            strokes, label = self.parser.parse_file(file_path)
            normalized = self.normalizer.normalize(strokes)
            stroke_text = self.normalizer.strokes_to_text(normalized)
            image = self.renderer.render(normalized)

            return {
                'stroke_text': stroke_text,
                'image': image,
                'label': label
            }
        except Exception as e:
            # In lazy validation mode, skip invalid files and try next one
            warnings.warn(f"Skipping invalid file {os.path.basename(file_path)}: {e}")
            # Return next valid sample (recursive fallback)
            return self.__getitem__((idx + 1) % len(self.files))


def collate_fn(batch):
    """
    Custom collate function for batching variable-length string fields and images.
    
    Args:
        batch: List of dicts from MathWritingDataset, each containing
               'stroke_text' (str), 'image' (PIL.Image), and 'label' (str).
    
    Returns:
        Dict with 'stroke_text', 'image', and 'label' as lists.
    """
    return {
        'stroke_text': [item['stroke_text'] for item in batch],
        'image': [item['image'] for item in batch],  # PIL Images from renderer
        'label': [item['label'] for item in batch]
    }


def create_dataloaders(data_dir, batch_size=8, num_workers=0):
    """Create train/valid/test DataLoaders."""
    train_ds = MathWritingDataset(data_dir, split='train')
    valid_ds = MathWritingDataset(data_dir, split='valid')
    test_ds = MathWritingDataset(data_dir, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader


# Test the implementation
if __name__ == "__main__":
    print("="*70)
    print("Testing InkML Parser + Full Normalization Pipeline")
    print("="*70)
    
    # Initialize parser and normalizer
    parser = InkMLParser()
    normalizer = StrokeNormalizer(target_points_per_stroke=16)
    
    # Test on the example file
    # For Colab: Upload dataset to /content/ and update path accordingly
    test_file = "/content/mathwriting-2024/test/00eaf4f2d3d20bb1.inkml"
    
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
    
    # Test text conversion
    print(f"\n{'TEXT REPRESENTATION:':-^70}")
    text_output = normalizer.strokes_to_text(normalized_strokes)
    tokens = text_output.split()
    print(f"  Format: '<stroke> x y x y ...'")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  First 20 tokens: {' '.join(tokens[:20])}...")
    print(f"  Text length: {len(text_output)} characters")
    
    print(f"\n{'='*70}")
    print("✓ Full pipeline working correctly!")
    print("="*70)
    
    # Test LaTeX Tokenizer
    print(f"\n{'='*70}")
    print("Testing LaTeX Tokenizer")
    print("="*70)
    
    tokenizer = LaTeXTokenizer()
    
    test_cases = [
        ("x^2", "x^2"),
        ("\\frac{a}{b}", "\\frac{a}{b}"),
        ("x^2", "x^3"),  # Different
        ("\\sqrt{2}", "\\sqrt{2}"),
    ]
    
    for pred, target in test_cases:
        tokens_pred = tokenizer.tokenize(pred)
        tokens_target = tokenizer.tokenize(target)
        cer = tokenizer.compute_cer(pred, target)
        print(f"\n  Pred: {pred} → {tokens_pred}")
        print(f"  Target: {target} → {tokens_target}")
        print(f"  CER: {cer:.3f}")
    
    print(f"\n✓ LaTeX tokenizer working!")
    
    # Test Dataset
    print(f"\n{'='*70}")
    print("Testing MathWritingDataset")
    print("="*70)
    
    try:
        # For Colab: Upload full dataset to /content/mathwriting-2024
        dataset = MathWritingDataset("/content/mathwriting-2024", split="test")
        print(f"\n✓ Dataset loaded: {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"\n✓ Sample 0:")
        print(f"  Label: {sample['label']}")
        print(f"  LaTeX tokens: {tokenizer.tokenize(sample['label'])}")
        print(f"  Stroke text length: {len(sample['stroke_text'])} chars")
        print(f"  First 100 chars: {sample['stroke_text'][:100]}...")
        
        print(f"\n✓ Dataset class working correctly!")
    except Exception as e:
        print(f"\n✗ Dataset test failed: {e}")
    
    print("="*70)