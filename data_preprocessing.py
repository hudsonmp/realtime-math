import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


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

# Parser has been tested and works.
if __name__ == "__main__":
    parser = InkMLParser()
    
    # Test on the example file
    test_file = "mathwriting-2024-excerpt/test/00eaf4f2d3d20bb1.inkml"
    strokes, label = parser.parse_file(test_file)
    
    print(f"Label: {label}")
    print(f"Number of strokes: {len(strokes)}")
    print(f"\nStroke details:")
    for i, stroke in enumerate(strokes):
        print(f"  Stroke {i}: {stroke.shape} - {len(stroke)} points")
        print(f"    First point: x={stroke[0,0]:.1f}, y={stroke[0,1]:.1f}, t={stroke[0,2]:.1f}")
        print(f"    Last point:  x={stroke[-1,0]:.1f}, y={stroke[-1,1]:.1f}, t={stroke[-1,2]:.1f}")