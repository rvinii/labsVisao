import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Caminho específico para a imagem fornecida
caminho_imagem = '/home/vini/Documentos/LaboratoriosVisao/lab04/images/pontos_ransac.png'

# Carrega a imagem em escala de cinza usando o caminho fornecido
imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print(f"Erro ao carregar a imagem: {caminho_imagem}")
    sys.exit()

# Função para ajustar uma reta aos pontos fornecidos, usando uma abordagem de regressão linear simples
def ajustar_reta(pontos):
    """Realiza o ajuste de uma linha y = mx + b aos dados."""
    # Separa as coordenadas x e y dos pontos
    x_vals = pontos[:, 0]
    y_vals = pontos[:, 1]

    # Calcula a média de x e y
    media_x = np.mean(x_vals)
    media_y = np.mean(y_vals)

    # Cálculo do numerador e denominador da equação para inclinação
    soma_produtos = np.sum((x_vals - media_x) * (y_vals - media_y))
    soma_quadrados = np.sum((x_vals - media_x) ** 2)

    # Verifica se todos os x são iguais, o que torna impossível ajustar uma linha
    if soma_quadrados == 0:
        print("Não é possível ajustar uma linha quando todos os valores de x são iguais.")
        return None
    
    # Calcula os coeficientes da reta: inclinação (m) e intercepto (b)
    inclinacao = soma_produtos / soma_quadrados
    intercepto = media_y - inclinacao * media_x

    return inclinacao, intercepto  # Retorna os parâmetros da linha ajustada

# Função para computar o erro absoluto entre um ponto e o modelo de reta ajustada
def calcular_erro(ponto, modelo):
    """Calcula a diferença entre o ponto fornecido e o valor previsto pelo modelo da reta."""
    m, b = modelo  # Extrai os coeficientes da reta
    x_val, y_val = ponto  # Extrai as coordenadas do ponto
    return abs(y_val - (m * x_val + b))  # Calcula o erro absoluto

# Implementação do algoritmo RANSAC para ajuste robusto de linhas com inliers e outliers
def ransac_line_fit(pontos, iteracoes, limite_erro, minimo_inliers):
    """Ajusta uma linha robusta usando o algoritmo RANSAC, lidando com outliers."""
    melhor_modelo = None  # Inicializa o melhor modelo como vazio
    melhores_inliers = []  # Inicializa a lista dos melhores inliers encontrados

    # Executa a quantidade de iterações especificada
    for _ in range(iteracoes):
        # Seleciona aleatoriamente dois pontos do conjunto
        amostra = random.sample(list(pontos), 2)
        modelo_reta = ajustar_reta(np.array(amostra))  # Ajusta uma reta a esses dois pontos

        # Se não foi possível ajustar a linha, continua para a próxima iteração
        if modelo_reta is None:
            continue

        inliers_correntes = []  # Lista de inliers da iteração atual

        # Avalia o erro de cada ponto em relação ao modelo
        for ponto in pontos:
            erro = calcular_erro(ponto, modelo_reta)
            if erro < limite_erro:  # Se o erro for menor que o limite, o ponto é um inlier
                inliers_correntes.append(ponto)

        # Atualiza o melhor modelo se o número de inliers for maior que o anterior e atingir o mínimo necessário
        if len(inliers_correntes) > len(melhores_inliers) and len(inliers_correntes) >= minimo_inliers:
            melhor_modelo = modelo_reta
            melhores_inliers = inliers_correntes

    return melhor_modelo, melhores_inliers  # Retorna o melhor modelo e seus inliers

# Aplica o detector de bordas de Canny para encontrar as bordas na imagem
bordas = cv2.Canny(imagem, 50, 150)

# Extrai as coordenadas dos pixels onde as bordas foram detectadas
pontos_bordas = np.argwhere(bordas > 0)
pontos_bordas = np.flip(pontos_bordas, axis=1)  # Inverte para obter as coordenadas no formato (x, y)

# Configura os parâmetros do algoritmo RANSAC
total_iteracoes = 50
limite_erro = 2
min_inliers = 50

# Executa o ajuste robusto de uma linha com o algoritmo RANSAC
modelo_final, inliers_final = ransac_line_fit(pontos_bordas, total_iteracoes, limite_erro, min_inliers)

# Calcula o número de inliers e outliers
total_inliers = len(inliers_final)
total_outliers = len(pontos_bordas) - total_inliers
print(f"Inliers: {total_inliers}, Outliers: {total_outliers}")

# Se um modelo válido foi encontrado, plota o resultado
if modelo_final is not None:
    m, b = modelo_final  # Extrai a inclinação e o intercepto da reta
    x_vals = np.array([0, imagem.shape[1]])  # Define o intervalo de x para a reta
    y_vals = m * x_vals + b  # Calcula os valores de y correspondentes

    # Exibe a imagem com a reta ajustada e os inliers
    plt.imshow(imagem, cmap='gray')
    plt.plot(x_vals, y_vals, color='red', label='Reta ajustada (RANSAC)')
    plt.scatter(np.array(inliers_final)[:, 0], np.array(inliers_final)[:, 1], color='green', s=1, label='Inliers')
    plt.legend()
    plt.show()
else:
    print("Nenhuma reta adequada foi encontrada.")
