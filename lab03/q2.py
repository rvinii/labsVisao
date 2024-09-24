import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Definir o caminho relativo da imagem
caminho_imagem = 'lab03/imagens/gradiente.png'

# Verifica se o caminho relativo existe
if not os.path.exists(caminho_imagem):
    print(f"O caminho {caminho_imagem} não foi encontrado. Verifique o diretório atual.")
    print(f"Diretório atual: {os.getcwd()}")
    sys.exit()

# Carrega a imagem em BGR
imagem_colorida = cv2.imread(caminho_imagem)

# Verifica se a imagem foi carregada corretamente
if imagem_colorida is None:
    print("Erro ao carregar a imagem. Verifique o caminho fornecido.")
    sys.exit()

# Função para converter imagem colorida em escala de cinza usando pesos diferentes para cada canal
def converter_para_cinza_alternativo(imagem):
    # Utiliza pesos personalizados para cada canal de cor (BGR)
    pesos = [0.114, 0.587, 0.299]  # Padrão invertido de OpenCV (BGR)
    imagem_cinza = np.dot(imagem[..., :3], pesos)
    return np.uint8(imagem_cinza)

# Aplica a conversão usando a nova função
imagem_convertida_alternativa = converter_para_cinza_alternativo(imagem_colorida)

# Função para exibir as imagens
def exibir_imagens_comparativas(img_original, img_convertida, titulo_convertida):
    plt.figure(figsize=(14, 7))

    # Imagem Original
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    # Imagem Convertida
    plt.subplot(1, 2, 2)
    plt.imshow(img_convertida, cmap='gray')
    plt.title(titulo_convertida)
    plt.axis('off')

    plt.show()

# Exibe a imagem original e a imagem convertida em escala de cinza
exibir_imagens_comparativas(imagem_colorida, imagem_convertida_alternativa, 'Imagem em Escala de Cinza (Método Alternativo)')
