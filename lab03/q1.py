import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para exibir imagens lado a lado para comparação
def exibir_imagens(titulos, imagens):
    plt.figure(figsize=(15, 5))
    for i in range(len(imagens)):
        plt.subplot(1, len(imagens), i+1)
        plt.title(titulos[i])
        plt.imshow(imagens[i], cmap='gray')
        plt.axis('off')
    plt.show()

# Função para aplicar filtros no domínio espacial e da frequência
def aplicar_filtros_alternativos(imagem_path, kernel_tamanho, raio_passabaixa):
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar filtro de mediana (domínio espacial) para redução de ruído sal e pimenta
    imagem_filtro_mediana = cv2.medianBlur(imagem, kernel_tamanho)

    # Transformada de Fourier para o domínio da frequência
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Criar um filtro passa-baixa em formato circular
    rows, cols = imagem.shape
    crow, ccol = rows // 2, cols // 2
    mascara = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    distancia = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mascara[distancia <= raio_passabaixa] = 1

    # Aplicar a máscara e transformar de volta para o domínio do espaço
    fshift = dft_shift * mascara
    img_back_shift = np.fft.ifftshift(fshift)
    imagem_freq_back = cv2.idft(img_back_shift)
    imagem_freq_back = cv2.magnitude(imagem_freq_back[:, :, 0], imagem_freq_back[:, :, 1])

    # Normalizar para visualização
    cv2.normalize(imagem_freq_back, imagem_freq_back, 0, 255, cv2.NORM_MINMAX)
    imagem_freq_back = np.uint8(imagem_freq_back)

    return imagem, imagem_filtro_mediana, imagem_freq_back

# Caminhos para as imagens
caminhos_imagens = [
    'imagens/salt_noise.png',
    'imagens/halftone.png',
    'imagens/pieces.png'
]

# Parâmetros dos filtros
tamanho_kernel = 5
raio_filtro = 30

# Aplicar filtros e exibir resultados
for caminho in caminhos_imagens:
    original, filtro_espacial, filtro_frequencia = aplicar_filtros_alternativos(caminho, tamanho_kernel, raio_filtro)

    # Exibir todas as imagens em uma única linha para comparação
    exibir_imagens(
        ["Imagem Original", f"Filtro de Mediana ({tamanho_kernel}x{tamanho_kernel})", "Filtro Passa-Baixa (Domínio da Frequência)"],
        [original, filtro_espacial, filtro_frequencia]
    )

