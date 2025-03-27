from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\yolo11m.pt')
    result = model.train(data="C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\Fuzzy---Project-2\\data.yaml", epochs=20, imgsz=640, device=0)