import cv2
import numpy as np
import sys

def swap_colors(image):
    # Convertendo a imagem para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definindo os intervalos de cor para azul
    lower_blue = np.array([77, 45, 26])
    upper_blue = np.array([108, 255, 255])

    # Criando uma mascara para a cor azul
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Definindo os intervalos de cor para verde
    lower_green = np.array([20, 25, 53])
    upper_green = np.array([75, 255, 255])

    # Criando uma mascara para a cor verde
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Substituindo a cor azul por um tom de verde
    hsv_image[mask_blue > 0, 0] = 47  # Altera apenas o componente H para um tom de verde

    # Substituindo a cor verde por um tom de azul
    hsv_image[mask_green > 0, 0] = 109  # Altera apenas o componente H para um tom de azul

    # Convertendo de volta para BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return result_image

# O código abaixo é para ler a imagem e salvar o resultado final após a troca de cores.
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py caminho_para_sua_imagem.jpg")
        sys.exit()

    filename = sys.argv[1]
    image = cv2.imread(filename)

    if image is None:
        print("Erro ao ler a imagem.")
        sys.exit()

    # Chamando a função de troca de cores
    result_image = swap_colors(image)

    # Salvar a imagem modificada
    output_filename = f"modificada_{filename}"
    cv2.imwrite(output_filename, result_image)
    print(f"A imagem modificada foi salva como: {output_filename}")
