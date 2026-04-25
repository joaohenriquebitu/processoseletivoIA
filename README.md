# Desafio MNIST: Classificação de Dígitos para Edge AI


* **Nome:** João Henrique de Brito Leandro Bitu Corrêa

---

## 1. Visão Geral do Projeto
Este projeto consiste no desenvolvimento, treinamento e otimização de uma Rede Neural Convolucional (CNN) para o reconhecimento de dígitos manuscritos (MNIST). O foco central foi desenvolver um modelo voltado para o Edge AI, buscando uma arquitetura que equilibre alta acurácia com baixo consumo de recursos, permitindo a execução em dispositivos de hardware limitado como microcontroladores e sistemas embarcados.

---

## 2. Treinamento e Arquitetura do Modelo (train_model.py)
Para atingir o nível máximo de eficiência, a arquitetura foi desenhada seguindo princípios de extratividade e simplicidade:

* **Arquitetura:** CNN com 3 camadas convolucionais.
    * **Layer 1 (Conv2D):** 32 filtros (3x3), ativação ReLU. Extração de padrões geométricos primários.
    * **Max Pooling:** Redução de dimensionalidade para diminuir o custo computacional das camadas seguintes.
    * **Layer 2 (Conv2D):** 64 filtros (3x3), ativação ReLU. Reconhecimento de formas complexas e curvas.
    * **Max Pooling:** Segunda redução espacial.
    * **Layer 3 (Conv2D):** 64 filtros (3x3), ativação ReLU. Refinamento de características para a classificação final.
    * **Flatten & Dense:** Camada densa com 64 neurônios para lógica final de classificação e camada de saída com Softmax para as 10 classes (0-9).
* **Justificativa:** O uso de MaxPooling entre as camadas garante que o modelo seja leve, processando apenas as informações essenciais e economizando ciclos de CPU, respeitando as restrições de tempo de execução do pipeline de integração contínua (CI).

---

## 3. Métricas de Resultado
O desempenho foi validado através de métricas claras e precisas:

* **Acurácia:** O modelo atingiu 99.08% no conjunto de testes tanto no modelo original (Keras) quanto no otimizado (TFLite).
* **Loss Function:** Utilizada a Sparse Categorical Crossentropy, ideal para classificação multiclasse de dígitos.
* **Batch Size & Epochs:** O treino foi configurado para 5 épocas, atingindo a convergência máxima sem sofrer overfitting e garantindo um modelo leve e com alto poder de generalização.

---

## 4. Geração do Modelo Treinado e Comparativo de Tamanho
Os artefatos foram salvos e otimizados, apresentando uma redução drástica no consumo de armazenamento:

* **Modelo Keras (model.h5):** 1,11 MB
* **Modelo TFLite (model.tflite):** 101 KB
* **Taxa de Compressão:** Redução de aproximadamente 91% no tamanho total do arquivo.

---

## 5. Conversão e Otimização para TFLite (optimize_model.py)
A etapa de conversão demonstra aprofundamento técnico em Edge AI através da aplicação de otimizações de pesos:

* **Técnica Utilizada:** Dynamic Range Quantization.
* **Trade-off (Tamanho x Desempenho):** Esta técnica converte os pesos de ponto flutuante que ocupam 32 bits na memória para inteiros que ocupam apenas 8 bits.
    * **Eficiência de Armazenamento:** A redução para 101 KB permite que o modelo seja armazenado em sistemas com memória Flash extremamente limitada.
    * **Impacto na Acurácia:** A perda de precisão foi inexistente neste projeto (99.08% mantidos em ambos os formatos), provando que a arquitetura proposta é resiliente à redução de precisão binária.

---

## 6. Validação Comparativa (validate_comparison.py)
Este script foi desenvolvido exclusivamente para validar e garantir a integridade do modelo após a otimização, não fazendo parte do processo de treino e otimização:
* **Acurácia do H5  (Float32):** 0.9908
* **Acurácia do TFLite  (Int8):** 0.9908
* **Veredito:** Sucesso na manutenção da inteligência do modelo com redução significativa da ocupação de memória e otimização para inferência até mesmo em CPU de baixa potência.

---

## 7. Como Executar o Projeto

1. **Ambiente:** Este projeto foi feito utilizando o VS Code Dev Container fornecido (Python 3.11).
2. **Dependências:**
    * Bibliotecas utilizadas:

        * TensorFlow: 2.21.0

        * NumPy: 2.4.4
          
    As bibliotecas podem ser instaladas por meio do comando:

   `pip install -r requirements.txt`
3. **Fluxo de Execução:**

    * *1. Treinar o modelo e gerar o arquivo model.h5*
    `python3 train_model.py`

    * *2. Otimizar o modelo para TFLite e gerar o arquivo model.tflite*
    `python3 optimize_model.py`

    * *3. Utilizar `python3 validate_comparison.py` caso queira validar e comparar as métricas entre o modelo original e o otimizado*
    
