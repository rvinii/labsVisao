import cv2
import numpy as np

# Carregar as imagens
circle_img = cv2.imread("/home/vini/Documentos/lab02 - Visão Computacional/Questão 02/circle.jpg")
line_img = cv2.imread("/home/vini/Documentos/lab02 - Visão Computacional/Questão 02/line.jpg")

# Criar uma nova tela em branco de 300x300 pixels
canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255  # Branco

# Definir a cor preta em BGR
black = (0, 0, 0)

# O tronco usará a linha como está, então o comprimento do tronco é a altura da imagem da linha
tronco_length = line_img.shape[0]

# O comprimento do braço será 75% do tronco
braço_length = int(0.75 * tronco_length)

# O comprimento da perna será o dobro do comprimento do braço
perna_length = 2 * braço_length

# Calcular o ponto central do tronco para centralizar o boneco
tronco_center_x = canvas.shape[1] // 2
tronco_start_y = canvas.shape[0] // 2 - tronco_length // 2

# Posicionar o tronco no centro da imagem
tronco_start_point = (tronco_center_x, tronco_start_y)
tronco_end_point = (tronco_center_x, tronco_start_y + tronco_length)
cv2.line(canvas, tronco_start_point, tronco_end_point, black, 3)

# Calcular o ponto inicial dos braços para estar no meio do tronco
braço_y = tronco_start_y + tronco_length // 4  # Colocando os braços um quarto abaixo do topo do tronco

# Desenhar os braços retos (horizontais)
cv2.line(canvas, (tronco_center_x - braço_length, braço_y), (tronco_center_x + braço_length, braço_y), black, 3)

# Desenhar as pernas
# Ângulo de 45 graus para cada perna em relação à vertical para obter 90 graus entre elas
cv2.line(canvas, tronco_end_point, (tronco_end_point[0] - perna_length // 2, tronco_end_point[1] + perna_length), black, 3)
cv2.line(canvas, tronco_end_point, (tronco_end_point[0] + perna_length // 2, tronco_end_point[1] + perna_length), black, 3)

# Desenhar a cabeça
# A cabeça será posicionada no topo do tronco
head_radius = circle_img.shape[0] // 2
head_center = (tronco_center_x, tronco_start_y - head_radius)
cv2.circle(canvas, head_center, head_radius, black, -1)

# Salvar a imagem resultante no sistema de arquivos
output_path = '/home/vini/Documentos/lab02 - Visão Computacional/Questão 02/boneco_palito.jpg'
cv2.imwrite(output_path, canvas)

# Mostrar o caminho onde a imagem foi salva
print(f"A imagem foi salva em: {output_path}")
