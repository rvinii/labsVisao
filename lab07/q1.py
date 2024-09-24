import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminhos fixos das imagens a serem processadas
path_img1 = '/home/vini/Documentos/LaboratoriosVisao/lab07/img/campus_quixada1.png'
path_img2 = '/home/vini/Documentos/LaboratoriosVisao/lab07/img/campus_quixada2.png'

####################################################
# Carregamento e Redimensionamento das Imagens
####################################################

# Carrega as imagens do disco usando OpenCV
image1 = cv2.imread(path_img1)
image2 = cv2.imread(path_img2)

# Redimensiona as imagens para reduzir o tempo de processamento
small_image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))
small_image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2))

# Converte as imagens para escala de cinza para melhor processamento de características
gray_image1 = cv2.cvtColor(small_image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(small_image2, cv2.COLOR_BGR2GRAY)

####################################################
# Detecção de Características com SIFT
####################################################

# Cria o objeto SIFT para detectar keypoints e calcular descritores
sift = cv2.SIFT_create()

# Detecta keypoints e calcula descritores para ambas as imagens
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

####################################################
# Correspondência de Características usando BFMatcher
####################################################

# Instancia o BFMatcher para comparar os descritores das duas imagens
matcher = cv2.BFMatcher()

# Encontra correspondências entre os descritores usando KNN (com k=2)
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Aplica o Teste de Razão para filtrar correspondências válidas
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

####################################################
# Verificação e Cálculo da Homografia
####################################################

# Verifica se há correspondências suficientes para calcular a homografia
if len(good_matches) >= 4:
    # Extrai as coordenadas dos keypoints correspondentes
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcula a matriz de homografia usando RANSAC para minimizar o impacto de outliers
    homography_matrix, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    ####################################################
    # Transformação e Combinação das Imagens
    ####################################################

    # Obtém as dimensões das imagens redimensionadas
    height1, width1 = small_image1.shape[:2]
    height2, width2 = small_image2.shape[:2]

    # Define os cantos da primeira imagem para aplicarmos a transformação de perspectiva
    corners = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    
    # Aplica a homografia aos cantos da primeira imagem para encontrar a nova posição deles
    transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)
    
    # Calcula os limites da nova imagem combinada
    min_x, min_y = np.int32(transformed_corners.min(axis=0).ravel())
    max_x, max_y = np.int32(transformed_corners.max(axis=0).ravel())
    max_x, max_y = max(max_x, width2), max(max_y, height2)

    # Calcula a translação necessária para colocar a imagem transformada dentro da área visível
    translation = [-min_x, -min_y]

    # Cria a matriz de translação
    translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Ajusta a homografia inicial com a matriz de translação para evitar cortar a imagem transformada
    adjusted_homography = translation_matrix.dot(homography_matrix)

    # Define o tamanho da imagem final combinada
    final_size = (max_x - min_x, max_y - min_y)

    # Aplica a transformação de perspectiva à primeira imagem
    transformed_image1 = cv2.warpPerspective(small_image1, adjusted_homography, final_size)

    # Sobrepõe a segunda imagem no espaço combinado, utilizando a translação calculada
    transformed_image1[translation[1]:translation[1] + height2, translation[0]:translation[0] + width2] = small_image2

    ####################################################
    # Exibição das Imagens
    ####################################################

    # Configura o layout para exibir as imagens lado a lado e a imagem combinada
    plt.figure(figsize=(18, 6))

    # Exibe a primeira imagem redimensionada
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(small_image1, cv2.COLOR_BGR2RGB))
    plt.title("Imagem 1 Redimensionada")
    plt.axis('off')

    # Exibe a segunda imagem redimensionada
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(small_image2, cv2.COLOR_BGR2RGB))
    plt.title("Imagem 2 Redimensionada")
    plt.axis('off')

    # Exibe a imagem final transformada e combinada
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(transformed_image1, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Combinada Transformada")
    plt.axis('off')

    # Mostra as imagens na tela
    plt.show()
else:
    # Lança um erro se não houver correspondências suficientes para calcular a homografia
    raise ValueError("Não há correspondências suficientes para calcular a homografia.")
