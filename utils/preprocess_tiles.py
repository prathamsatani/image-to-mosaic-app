import os
import pandas as pd
import cv2

class PreprocessTiles:
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
            "dominant_color": dominant_color
        }
    
    def preprocess_tiles(self, dir: str):
        tiles_metadata = []
        for filename in os.listdir(dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(dir, filename)
                metadata = self.generate_image_metadata(filepath)
                tiles_metadata.append(metadata)
        return pd.DataFrame(tiles_metadata)

if __name__ == "__main__":
    # processor = PreprocessTiles()
    # df = processor.preprocess_tiles(os.getcwd())
    print(os.getcwd())