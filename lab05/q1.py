import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a energia da imagem usando gradientes do OpenCV
def calculate_energy(image, mask):
    # Converter a imagem para escala de cinza e calcular os gradientes
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(grad_x**2 + grad_y**2)
    # Diminuir a energia dentro da máscara para priorizar a remoção do objeto
    energy[mask == 255] -= 1000
    return energy

# Função para encontrar a costura com menor energia
def find_seam(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(c):
            # Ajuste para evitar índices negativos e fora do intervalo
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            elif j == c - 1:
                idx = np.argmin(M[i-1, j-1:j+1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

# Função para remover a costura da imagem e da máscara
def remove_seam(image, backtrack, mask=None):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    new_mask = np.zeros((r, c - 1), dtype=mask.dtype) if mask is not None else None
    j = np.argmin(backtrack[-1])
    
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        if mask is not None:
            new_mask[i, :] = np.delete(mask[i, :], [j])
        j = backtrack[i, j]

    return output, new_mask

# Função principal que aplica o seam carving
def seam_carving(image, mask, num_seams):
    for _ in range(num_seams):
        print(f"Aplicando seam carving, passo {_+1} de {num_seams}...")
        energy = calculate_energy(image, mask)
        M, backtrack = find_seam(energy)
        image, mask = remove_seam(image, backtrack, mask)
    return image

# Função para criar uma máscara para o objeto a ser removido
def create_mask(image, lower_color, upper_color):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_color, upper_color)
    return mask

# Caminho para a imagem de entrada
img_path = '/home/vini/Documentos/LaboratoriosVisao/lab05/galinha.png'
print("Carregando a imagem...")
img = cv2.imread(img_path)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro ao carregar a imagem: {img_path}")
else:
    print("Imagem carregada com sucesso!")

    # Definir o intervalo de cores para a máscara (vermelho)
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])

    print("Criando a máscara...")
    # Criar a máscara para o ponto vermelho
    mask = create_mask(img, lower_color, upper_color)
    print("Máscara criada!")

    # Mostrar a máscara para garantir que está correta
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara para Remoção')
    plt.axis('off')
    plt.show()

    # Mostrar o mapa de energia
    print("Calculando a energia...")
    energy = calculate_energy(img, mask)
    plt.imshow(energy, cmap='gray')
    plt.title('Mapa de Energia')
    plt.axis('off')
    plt.show()

    # Calcular quantas costuras remover com base no tamanho do objeto
    num_seams = 100

    print("Iniciando o seam carving...")
    # Aplicar o seam carving para remover o ponto vermelho
    result_image = seam_carving(img, mask, num_seams)

    print("Seam carving concluído!")
    # Caminho para salvar a imagem resultante
    output_path = '/home/vini/Documentos/LaboratoriosVisao/lab05/galinha_sem_ponto.png'

    # Salvar a imagem modificada
    cv2.imwrite(output_path, result_image)
    print(f"Imagem salva com sucesso em: {output_path}")

    # Mostrar a imagem original e a imagem modificada
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Sem Ponto Vermelho')
    plt.axis('off')

    plt.show()
