import cv2
import numpy as np
from util import get_limits  # Importa la función get_limits desde util.py

# Cargar el clasificador pre-entrenado para la detección de carros
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

cap = cv2.VideoCapture(0)  # Usar la cámara en vivo (puedes cambiar el índice si tienes varias cámaras)

while True:
    ret, frame = cap.read()  # Capturar un frame de la cámara
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises para la detección

    # Detectar carros en el frame con parámetros específicos para mejorar la detección
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Mostrar solo el color del carro detectado
    for (x, y, w, h) in cars:
        roi = frame[y:y + h, x:x + w]  # Región de interés del automóvil
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Convertir ROI a espacio de color HSV

        # Escoger un color específico del carro
        color = hsv_roi[int(h/2), int(w/2)]  # Obtener el color del centro del carro (puede ajustarse)

        lower_limit, upper_limit = get_limits(color)  # Obtener los límites del color

        # Crear una máscara para el color del carro
        mask = cv2.inRange(hsv_roi, lower_limit, upper_limit)

        # Convertir la máscara a BGR para obtener un color visible
        color_bgr = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0]

        # Dibujar un rectángulo con el color del carro alrededor del carro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), 2)

    cv2.imshow('Car Detection', frame)  # Mostrar el frame con la detección de carros

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir del bucle
        break

cap.release()
cv2.destroyAllWindows()
