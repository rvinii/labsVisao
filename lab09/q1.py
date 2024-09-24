import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import urllib
import os

# Função para calcular o IoU entre dois bounding boxes
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Função para detectar rostos usando Haar Cascade
def detectar_haar(image, model_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(model_path)
    results = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
    return results

# Função para detectar rostos usando YOLO
def detectar_yolo(image, model_cfg, model_weights, class_names):
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    layer_names = net.getLayerNames()
    # Ajuste aqui
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Preprocessamento da imagem para YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    height, width = image.shape[:2]
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] == 'person':  # Detectar pessoas como rostos
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    # Aplica Non-Maxima Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = [boxes[i] for i in indices.flatten()]
    return detections

# Função para detectar rostos usando RetinaFace
def detectar_retinaface(image):
    faces = RetinaFace.detect_faces(image)
    detections = []
    for key in faces:
        face = faces[key]
        facial_area = face['facial_area']
        detections.append(facial_area)
    return detections

# Função para baixar arquivos do modelo
def baixar_modelo(url, nome_arquivo):
    if not os.path.exists(nome_arquivo):
        urllib.request.urlretrieve(url, nome_arquivo)
        print(f"Modelo {nome_arquivo} baixado com sucesso!")
    else:
        print(f"Modelo {nome_arquivo} já existe.")

# Caminho da imagem e dos modelos
image_path = '/home/vini/Documentos/LaboratoriosVisao/lab09/imagens/testeface.jpg' # Altere para o caminho correto da imagem no Colab
haar_model_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
yolo_cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg"
yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
class_names_url = "https://github.com/pjreddie/darknet/raw/master/data/coco.names"

# Baixar os modelos
baixar_modelo(haar_model_url, "haarcascade_frontalface_default.xml")
baixar_modelo(yolo_cfg_url, "yolov3.cfg")
baixar_modelo(yolo_weights_url, "yolov3.weights")
baixar_modelo(class_names_url, "coco.names")

haar_model_path = "haarcascade_frontalface_default.xml"
yolo_cfg = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
class_names_path = 'coco.names'

# Carrega a imagem
image = cv2.imread(image_path)
iou_values = []

# Detecção com Haar Cascade
haar_detections = detectar_haar(image, haar_model_path)
print(f"Detecções Haar: {len(haar_detections)}")
# Desenhar detecções na imagem
for (x, y, w, h) in haar_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Detecção com YOLO
with open(class_names_path, 'r') as f:
    class_names = f.read().strip().split('\n')
yolo_detections = detectar_yolo(image, yolo_cfg, yolo_weights, class_names)
print(f"Detecções YOLO: {len(yolo_detections)}")
# Desenhar detecções na imagem
for (x, y, w, h) in yolo_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Detecção com RetinaFace
retinaface_detections = detectar_retinaface(image)
print(f"Detecções RetinaFace: {len(retinaface_detections)}")
# Desenhar detecções na imagem
for (x, y, w, h) in retinaface_detections:
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

# Mostrar imagem final com as detecções de cada método
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detecções - Haar (Vermelho), YOLO (Azul), RetinaFace (Verde)')
plt.axis('off')
plt.show()

# Calcular IoU entre os métodos
for i in range(len(haar_detections)):
    if i < len(yolo_detections) and i < len(retinaface_detections):
        iou_haar_yolo = calcular_iou(haar_detections[i], yolo_detections[i])
        iou_haar_retinaface = calcular_iou(haar_detections[i], retinaface_detections[i])
        iou_yolo_retinaface = calcular_iou(yolo_detections[i], retinaface_detections[i])
        iou_values.extend([iou_haar_yolo, iou_haar_retinaface, iou_yolo_retinaface])

# Mostrar valores médios de IoU
if iou_values:
    print(f"Valor médio de IoU: {np.mean(iou_values)}")
else:
    print("Não foi possível calcular IoU para as detecções.")
