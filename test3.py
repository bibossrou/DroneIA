from ultralytics import YOLO
model = YOLO('C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\runs\\detect\\train7\\weights\\best.pt')  # charger un modèle personnalisé

# Prédire avec le modèle
results = model('C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\dataset\\train\\images\\1_4_jpeg.rf.702a4ebffae995d1c57c8383f4421d47.jpg')  # prédire sur une image