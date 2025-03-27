import cv2
import torch
from ultralytics import YOLO

# Charger le modèle YOLO (assure-toi d'avoir téléchargé le bon modèle .pt)
model = YOLO("C:\\Users\\bibos\\runs\\detect\\train23\\weights\\best.pt")  # Remplace par le chemin de ton modèle si différent

def main():
    cap = cv2.VideoCapture(0)  # Ouvre la webcam (0 pour la webcam principale)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire la frame.")
            break
        
        # Exécuter l'inférence avec YOLO
        results = model(frame)
        
        # Dessiner les résultats sur l'image
        annotated_frame = results[0].plot()
        
        # Afficher le flux vidéo avec détection
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)
        
        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
