from ultralytics import YOLO
import multiprocessing
import torch

torch.cuda.empty_cache()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Assurez-vous que CUDA est disponible
    if torch.cuda.is_available():
        print("CUDA est disponible, utilisation de GPU pour l'entraînement.")
        device = 'cuda'
    else:
        print("CUDA n'est pas disponible, utilisation du CPU pour l'entraînement.")
        device = 'cpu'

    model = YOLO('yolo11m.pt')
    result = model.train(data="dataset\data.yaml", epochs=1, imgsz=224,amp=False,batch=10, project="C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\YoloResult")
