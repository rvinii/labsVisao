import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho fixo da imagem
caminho_arquivo = '/home/vini/Documentos/LaboratoriosVisao/lab03/imagens/Pedra_galinha_choca_quixada_ce.jpeg'

# Verifica se o arquivo existe antes de tentar carregá-lo
if not os.path.exists(caminho_arquivo):
    print(f"Erro: O caminho especificado '{caminho_arquivo}' não existe.")
    sys.exit()

# Lê a imagem a partir do caminho fornecido em escala de cores BGR
imagem_bgr = cv2.imread(caminho_arquivo)

# Verifica se a imagem foi carregada corretamente
if imagem_bgr is None:
    print("Erro ao carregar a imagem. Verifique o caminho e tente novamente.")
    sys.exit()

# Função para aplicar efeito sépia com coeficientes ajustados
def aplicar_sepia_ajustado(imagem):
    # Normaliza os valores de pixel para o intervalo [0, 1]
    imagem_normalizada = imagem.astype(np.float32) / 255.0
    
    # Coeficientes ajustados para transformação sépia
    sepia_coefs = [
        [0.272, 0.534, 0.131],  # Peso para o canal B
        [0.349, 0.686, 0.168],  # Peso para o canal G
        [0.393, 0.769, 0.189]   # Peso para o canal R
    ]
    
    # Converte a matriz de coeficientes para numpy array e aplica na imagem
    matriz_sepia = np.array(sepia_coefs).T
    imagem_sepia = cv2.transform(imagem_normalizada, matriz_sepia)
    
    # Limita os valores para o intervalo [0, 1] para evitar estouros de cor
    imagem_sepia = np.clip(imagem_sepia, 0, 1)
    
    # Converte de volta para uint8
    return (imagem_sepia * 255).astype(np.uint8)

# Aplica a transformação sépia ajustada à imagem
imagem_sepia = aplicar_sepia_ajustado(imagem_bgr)

# Função para exibir a comparação lado a lado
def exibir_comparacao(imagem_original, imagem_transformada, titulo_transformada):
    plt.figure(figsize=(14, 7))
    
    # Subplot para a imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    # Subplot para a imagem transformada
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imagem_transformada, cv2.COLOR_BGR2RGB))
    plt.title(titulo_transformada)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Verifica as dimensões da imagem e informa o status
print(f"Imagem carregada com sucesso! Dimensões: {imagem_bgr.shape}")

# Exibe a imagem original e a imagem com filtro sépia lado a lado
exibir_comparacao(imagem_bgr, imagem_sepia, 'Imagem com Filtro Sépia (Coeficientes Ajustados)')
