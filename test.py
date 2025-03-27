from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO('C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\runs\\detect\\train18\\weights\\best.pt')

image = "C:\\Users\\bibos\\OneDrive\\Documents\\TRAVAIL\\IPSA\\scrypt\\DRONE + MODELE\\Drone_\\dataset\\train\\images\\1-70-_jpeg.rf.f09fd557fa16c4aaaf6aae3c2bda6cbf.jpg"
results = model.predict(image)
from PIL import Image
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image