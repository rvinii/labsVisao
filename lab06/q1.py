import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def match_surf(im1, im2):
    # Inicializa o detector SURF com um valor de Hessian Threshold (mínimo para detectar keypoints)
    minHessian = 400
    surf = cv.xfeatures2d_SURF.create(minHessian)

    # Detecta os keypoints (pontos de interesse) e calcula os descritores (informações locais)
    kp1, des1 = surf.detectAndCompute(im1, None)
    kp2, des2 = surf.detectAndCompute(im2, None)

    # Cria um objeto BFMatcher para encontrar correspondências entre os descritores das duas imagens
    bf = cv.BFMatcher()

    # Faz a correspondência usando o método KNN, que encontra os dois melhores matches (k=2) para cada descritor
    matches = bf.knnMatch(des1, des2, k=2)
    print("(SURF) Number of matches: ", len(matches))

    # Lista para armazenar as boas correspondências após o ratio test
    good = []

    # Ratio test: Compara o melhor match com o segundo melhor
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Teste de razão proposto por David Lowe
            good.append([m])  # Adiciona à lista 'good' os matches que passam no teste

    # Desenha os primeiros 15 matches bons
    img_match = cv.drawMatchesKnn(im1, kp1, im2, kp2, good[:15], None,
                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_match  # Retorna a imagem com as correspondências desenhadas

def matching_orb(im1, im2):
    # Inicializa o detector ORB, que é eficiente e rápido
    orb = cv.ORB_create()

    # Detecta os keypoints (pontos de interesse) e calcula os descritores com ORB
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    # Cria um objeto BFMatcher para encontrar correspondências
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Realiza a correspondência usando o BFMatcher
    matches = bf.match(des1, des2)
    print("(ORB) Number of matches: ", len(matches))

    # Ordena as correspondências com base na distância
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenha os primeiros 15 matches
    img_match = cv.drawMatches(im1, kp1, im2, kp2, matches[:15], None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_match  # Retorna a imagem com as correspondências desenhadas

# Caminhos das imagens fixos
img1_path = '/home/vini/Documentos/LaboratoriosVisao/lab06/images/antartica_lata.jpg'
img2_path = '/home/vini/Documentos/LaboratoriosVisao/lab06/images/antartica.jpg'

# Carrega as duas imagens em escala de cinza para processamento
img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

# Verifica se as imagens foram carregadas corretamente
if img1 is None or img2 is None:
    print("Erro ao carregar as imagens. Verifique os caminhos fornecidos.")
    sys.exit()

# Aplica a função match_surf() para encontrar correspondências com o método SURF
img_match_surf = match_surf(img1, img2)

# Aplica a função matching_orb() para encontrar correspondências com o método ORB
img_match_orb = matching_orb(img1, img2)

# Usamos matplotlib para exibir as imagens resultantes das correspondências lado a lado
plt.figure(figsize=(20, 10))

# Mostra os resultados da correspondência usando SURF
plt.subplot(121)  # Cria o primeiro subplot (esquerda)
plt.imshow(img_match_surf)  # Mostra a imagem resultante de SURF
plt.title('SURF')  # Título da imagem

# Mostra os resultados da correspondência usando ORB
plt.subplot(122)  # Cria o segundo subplot (direita)
plt.imshow(img_match_orb)  # Mostra a imagem resultante de ORB
plt.title('ORB')  # Título da imagem

# Exibe as imagens na tela
plt.show()
