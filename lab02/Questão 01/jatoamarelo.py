import cv2
import numpy as np

# Função para ajustar os canais de cor usando multiplicação simples, que é uma forma de transformação gamma
def adjust_colors(image, red_multiplier, green_multiplier, blue_multiplier):
    
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_channel = cv2.multiply(red_channel, red_multiplier)
    green_channel = cv2.multiply(green_channel, green_multiplier)
    blue_channel = cv2.multiply(blue_channel, blue_multiplier)

    # Merge the channels back together
    return cv2.merge([blue_channel, green_channel, red_channel])

# Carregar a imagem original
img_original = cv2.imread('/home/vini/Documentos/lab02 - Visão Computacional/Questão 01/jato.jpg')

# Aumentar os canais vermelho e verde, e diminuir o azul
img_amarelada = adjust_colors(img_original, 1.2, 1.2, 0.8)

# Salvar a imagem amarelada
output_path = '/home/vini/Documentos/lab02 - Visão Computacional/Questão 01/jato_amarelada.jpg'
cv2.imwrite(output_path, img_amarelada)
output_path
