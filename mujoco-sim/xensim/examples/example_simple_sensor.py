import cv2
import time
import numpy as np
from pathlib import Path

from xensim.core import SimpleSensor
from xensim import PROJ_DIR


sensor = SimpleSensor(PROJ_DIR / "assets/fem/g1-ws_table.npz", visible=True)

def generate_zmap(t):
    delta = t % 100
    x = np.linspace(-10, 10, 50, dtype='f4')
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing='xy')
    Z = -(np.sin(np.sqrt(X**2/4 + Y**2) * np.pi / 5 - delta*np.pi) + 1)
    return Z

def main():
    while True:
        sensor.step(generate_zmap(time.time()), smooth_norm=2, smooth_mesh=2)
        sensor.update()
        
        # img = img[..., ::-1]  # RGB to BGR
        # cv2.imshow("img", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    main()