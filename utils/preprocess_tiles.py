import os
import pandas as pd
import cv2

class PreprocessTiles:
    def _rgb_to_text(self, r, g, b):
        color = None
        if r > g and r > b:
            color = "Red"
        elif g > r and g > b:
            color = "Green"
        elif b > r and b > g:
            color = "Blue"
        elif r == g and r > b:
            color = "Yellow"
        elif r == b and r > g:
            color = "Cyan"
        elif g == b and g > r:
            color = "Magenta"
        return color

    def generate_image_metadata(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        height, width, channels = image.shape
        mode = "RGB" if channels == 3 else "Grayscale" if channels == 1 else "RGBA" if channels == 4 else "Unknown"
        format = os.path.splitext(image_path)[1].replace('.', '').upper()
        dominant_color = cv2.mean(image)[:3]
        
        return {
            "filename": os.path.basename(image_path),
            "width": width,
            "height": height,
            "mode": mode,
            "format": format,
            "average-red": dominant_color[2],
            "average-green": dominant_color[1],
            "average-blue": dominant_color[0],
            "dominant-color": self._rgb_to_text(int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
        }
    
    def preprocess_tiles(self, dir: str):
        tiles_metadata = []
        for filename in os.listdir(dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(dir, filename)
                metadata = self.generate_image_metadata(filepath)
                tiles_metadata.append(metadata)
                
        pd.DataFrame(tiles_metadata).to_csv("tiles_metadata.csv", index=False)
        print(f"Tiles metadata saved to {os.path.abspath('tiles_metadata.csv')}")

if __name__ == "__main__":
    processor = PreprocessTiles()
    processor.preprocess_tiles(os.path.join(os.getcwd(), "images"))