import cv2 as cv
import numpy as np
import math
import os
import csv

#Aquí la variable SZ se refiere al tamaño de la imagen a la que se le van a calclar los descriptores
SZ=256

def read_csv_file(file_path) -> list:
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        row_list = []
        for row in csv_reader:
            row_list.append(row)
        return row_list

def write_matrix_to_csv(matrix, file_path):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix)

# esta función encuentra lineas rectas de cierto tamaño mínimo
def find_lines(img):
	dst = cv.Canny(img, 120, 240, None, 3)
	linesP = cv.HoughLinesP(dst,1, np.pi / 180, 50, None, SZ/5, 10)
	return linesP

#Esta función detecta puntos de interés, que usualmente coiniden con las esquinas
def goodFeaturesToTrack(img): #img must be grayscale, val es el número de esquinas a rastrear
    # Parámetros para el algoritmo de (Shi-Tomasi algorithm)
    maxCorners = 100
    qualityLevel = 0.1
    minDistance = 8
    blockSize = 5
    gradientSize = 5
    useHarrisDetector = False
    k = 0.04
    # Se aplica con función de opencv
    corners = cv.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)	
    #print(" Número de esquinas detectadas:", corners.shape[0]) # comenta esta linea si quieres menos líneas en pantalla
    return corners 
   

# A partir de aquí se empieza a trabajar con la imagen
csv_file_path = 'card_evaluation.csv'
row_list = read_csv_file(csv_file_path)

for image in row_list:
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", image[0])
    src = cv.imread(image_dir)

    if src is None:
        print("Could not open or find the image")

    #ajustamos tamaño modelo de color
    src = cv.resize(src, (SZ, SZ))
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #usamos las funciones de arriba para detectar puntos de interes y líneas
    corners = goodFeaturesToTrack(src_gray)
    lines = find_lines(src_gray)

    #Generar los descriptores, el que está a contiuación es un histograma que guarda la frecuencia con que aparecen pixeles que son parte de una línea en cierta dirección. Se consideran 8 direcciónes, que corresponden a partir en 8 la inclinación de 0 a 180 grados. 
    hist = [0,0,0,0,0,0,0,0]
    total = 0
    for i in range(0, len(lines)):
        l = lines[i][0]
        x=int(l[2]-l[0])
        y=int(l[3]-l[1])
        
        mag=cv.cartToPolar(x, y)[0][0][0]
        ang=cv.cartToPolar(x, y)[1][0][0]
        bin_ang=int(np.round(ang/(2*np.pi)*16))
        
        if bin_ang>7:
            bin_ang= bin_ang-8
            
        if bin_ang ==8:
            bin_ang= bin_ang-8
            
        hist[bin_ang]+=int(mag)
        total+=int(mag)

    # se convierte el histograma para que sea con valores de 0 a 100, o sea, porcentual		
    for i in range(len(hist)):
        hist[i]=int(100*hist[i]/total)

    #El siguiente descriptor consiste en partir la imagen en 16 cuadrantes y contar cuántos puntos de interés hay en cada uno.
    corners_location=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(corners.shape[0]):
        x=int(corners[i,0,0])
        y=int(corners[i,0,1])
        cuadx= int(x/(SZ/4))
        cuady=  int(y/(SZ/4))
        corners_location[4*cuadx+cuady]+=1

    del image[0]
    image[-1:-1] = hist
    image[-1:-1] = corners_location
    
write_matrix_to_csv(row_list, "card_evaluation_result.csv")

#A partir de aquí lo que tienen que hacer es que los procesos de arriba se apliquen para toda su base de datos de train y toda su base de datos de test. Y que se guarde un archivo csv con los valores de los vectores que se obtienen y agregar al final si la imagen es de tarjeta o no. En este caso recomiendo poner 0 si no es tarjeta y uno si sí lo es. 
#Por ejemplo, vamos a suponer que las primeras son 3 imágenes de mi base da datos son 2 tarjeta y uno que no es. 
#La primera imagen, si corro este programa así como está obtiene: 
#[67, 0, 0, 6, 25, 0, 0, 0]
#[0, 5, 8, 0, 0, 8, 11, 0, 0, 6, 5, 1, 0, 0, 0, 1]
#La segunda imagen 
#[72, 0, 0, 0, 27, 0, 0, 0]
#[1, 7, 6, 5, 2, 6, 13, 6, 4, 6, 9, 2, 8, 6, 10, 2]
#Y la tercera, la que no es tarjeta, da: 
#[31, 3, 8, 10, 25, 1, 2, 16]
#[10, 14, 13, 4, 11, 9, 6, 3, 4, 9, 1, 0, 4, 11, 0, 1]

#Entonces tengo que hacer un programa que guarde en el archivo csv lo siguiente
'''
67, 0, 0,  6,  25, 0, 0, 0, 0,  5,  8,  0, 0,  8, 11, 0, 0, 6, 5, 1, 0, 0,  0, 1, 1
72, 0, 0,  0,  27, 0, 0, 0, 1,  7,  6,  5, 2,  6, 13, 6, 4, 6, 9, 2, 8, 6, 10, 2,1
31, 3, 8, 10, 25, 1, 2,16,10,14,13, 4,11, 9, 6,  3, 4, 9, 1, 0, 4, 11, 0, 1, 0
'''
#y así, pero con tantas filas como imágenes en la carpeta. 

# Recomiendo borrar todo lo que está abajo de este comentario porque es para visualizar, pero si se va a aplicar a muchas imágenes, mejor lo quitan. 
'''
copy = np.copy(src)
for i in range(corners.shape[0]):
	cv.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), 4, (250, 0, 0), cv.FILLED)

if lines is not None:
	for i in range(0, len(lines)):
		l = lines[i][0]
		cv.line(copy, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)

cv.namedWindow("Image")
cv.imshow("Image", copy)
cv.waitKey()
'''

