# Importar librerías
import cv2
import os
import numpy as np

# El móduldo LBPH se utiliza para reconocer rostros
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cascade que posee los patrones de los rostros de gatos
rostro = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')


def get_images_and_labels(path):
    # Se agregan todas las imágenes en una lista image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images contiene las imágenes de las caras de gatos
    images = []
    # labels contiene las etiquetas asignadas a cada imagen
    labels = []
    for image_path in image_paths:
        # Se lee la imagen y se convierte a escala de grises
        image_pil = cv2.imread(image_path, 0)
        # Se imprime el path de la imagen porque si
        print("Leyendo: "+image_path)
        # Se convierte el formato de la imagen a un array
        image = np.array(image_pil, 'uint8')

        # Se asignan las etiquetas de las imágenes
        if "persa" in os.path.split(image_path)[1]:
            nbr = 1
        elif "ScottishFold" in os.path.split(image_path)[1]:
            nbr = 2
        elif "siames" in os.path.split(image_path)[1]:
            nbr = 3
        elif "Sphynx" in os.path.split(image_path)[1]:
            nbr = 4
        elif "toyger" in os.path.split(image_path)[1]:
            nbr = 5
        elif "azul" in os.path.split(image_path)[1]:
            nbr = 6
        else:
            nbr = 0
        # Detectamos si hay un gato en la imagen
        faces = rostro.detectMultiScale(image)
        # Si el rostro es detectado se agrega éste a images y la etiqueta a labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
    # Retorna las listas images y labels
    return images, labels


# Ruta a la base de datos de entrenamiento
path = 'pro_entreno'
# Llamado a la función get_images_and_labels para obtener las imágenes de los rostros de gatos y las
# etiquetas correspondientes
images, labels = get_images_and_labels(path)
# Realizar el entrenamiento
recognizer.train(images, np.array(labels))
# Ruta a la base de datos de prueba
path2 = 'pro_identificar'
# Se agregan las imágenes de prueba (con extension _d) a la lista image_paths
image_paths = [os.path.join(path2, f) for f in os.listdir(path2)]
k=0
for image_path in image_paths:
    print("Comparando imagen: "+image_path)
    predict_image_pil = cv2.imread(image_path, 0)
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = rostro.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        k = k + 1
        nbr_predicted, conf = recognizer.predict(
            predict_image[y: y + h, x: x + w])
        # Se compara el label del reconocimiento y se asigna un valor
        if nbr_predicted == 1:
            n = ("PERSA")
        elif nbr_predicted == 2:
            n = ("SCOTTISHFOLD")
        elif nbr_predicted == 3:
            n = ("SIAMES")
        elif nbr_predicted == 4:
            n = ("SPHYNX")
        elif nbr_predicted == 5:
            n = ("TOYGER")
        elif nbr_predicted == 6:
            n = ("AZUL")
        else:
            n = ("NO RECONOCIDO")
        cv2.rectangle(predict_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(predict_image, "GATO "+n,
                    (x, y+h+20), 0, 0.55, (0, 255, 0), 2)
        cv2.imwrite(n+repr(k)+".jpg", predict_image)
        cv2.imshow("Gato reconocido", predict_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
