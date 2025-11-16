import random
import numpy as np
import cv2

class MosaicGenerator:
    def convert_to_chunks(self, img: np.ndarray, n: int = 3) -> list[np.ndarray]:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        dims = img.shape
        if n > dims[0]*dims[1]:
            raise ValueError("'n' should be less than lower dimension")
        
        chunks = []
        for j in range(n):
            row = []
            for k in range(n):
                row.append(np.array([img[i][int(dims[1]*k/n):int(dims[1]*(k+1)/n)] for i in range(int(len(img)*j/n), int(len(img)*(j+1)/n))]))
            chunks.append(row)
        return chunks

    def rotate_matrix(self, mat: np.ndarray, times: int = 1):
        arr = np.asarray(mat)
        dims = arr.shape
        if times % 4 == 0:
            return arr
        elif times % 4 == 1:
            return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE).reshape(dims)
        elif times % 4 == 2:
            return cv2.rotate(arr, cv2.ROTATE_180).reshape(dims)
        elif times % 4 == 3:
            return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(dims)

            
        return arr.reshape(dims)

    def stitch_chunks(self, chunks: list[list[np.ndarray]]) -> np.ndarray:
        rows = []
        for i in range(len(chunks)):
            rows.append(np.hstack(chunks[i]))
        
        return np.vstack(rows)

    def create_mosaic(self, image_array: np.ndarray, n_chunks: int) -> np.ndarray:
        chunks = self.convert_to_chunks(image_array, n_chunks) 
        for i in range(len(chunks)):
            for j in range(len(chunks[0])):
                t = random.randint(0, 10)
                chunks[i][j] = self.rotate_matrix(chunks[i][j], t)
        return self.stitch_chunks(chunks)    # type: ignore