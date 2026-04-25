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
      
O uso de MaxPooling entre as camadas garante que o modelo seja leve, processando apenas as informações essenciais e economizando ciclos de CPU.

---

## 3. Métricas de Resultado
O desempenho foi validado através das seguintes métricas:

* **Acurácia:** O modelo atingiu 99.08% no conjunto de testes tanto no modelo original (Keras) quanto no otimizado (TFLite).
* **Loss Function:** Utilizada a Sparse Categorical Crossentropy, ideal para classificação multiclasse de dígitos.
* **Batch Size & Epochs:** O treino foi configurado para 5 épocas, atingindo a convergência máxima sem sofrer overfitting e garantindo um modelo leve e com alto poder de generalização.

---

## 4. Conversão e Otimização para TFLite (optimize_model.py)
A etapa de conversão objetiva adaptar e compactar o modelo de IA para que ele seja compatível com as restrições de hardware dos dispositivos de borda.

**Técnicas Utilizadas:**
* **Dynamic Range Quantization:** Esta técnica converte os pesos de ponto flutuante que ocupam 32 bits na memória para inteiros que ocupam apenas 8 bits.
* **Constant Folding:** Simplificação da topologia da rede ao pré-calcular operações constantes e fundir camadas redundantes para otimizar a execução na CPU.

**Trade-off (Eficiência x Acurácia):**

* **Eficiência:** A redução para pesos de 8 bits permite que o modelo seja armazenado em sistemas com memória Flash extremamente limitada, além de que cálculos com inteiros exigem menos ciclos de CPU. Somado a isso, o *Constant Folding* reduz o overhead de processamento ao simplificar o grafo da rede, eliminando operações redundantes e acelerando a inferência.
* **Impacto na Acurácia:** A perda de precisão foi inexistente neste projeto (99.08% mantidos em ambos os formatos). Além disso, como o *Constant Folding* lida apenas com a simplificação matemática de operações constantes sem alterar os pesos aprendidos, ele garante uma execução mais rápida sem qualquer degradação na inteligência do modelo.

---

## 5. Validação Comparativa (validate_comparison.py)
Este script valida a integridade do modelo após a otimização, comparando o desempenho sob restrições de hardware:

### Comparativo de Acurácia
* **Acurácia do H5 (Float32):** 0.9908
* **Acurácia do TFLite (Int8):** 0.9908
* **Resultado:** A precisão foi mantida integralmente, provando que a quantização não degradou a inteligência do modelo.

### Comparativo de Tamanho (Armazenamento)
* **Modelo Keras (model.h5):** 1,11 MB
* **Modelo TFLite (model.tflite):** 101 KB
* **Taxa de Compressão:** Redução de aproximadamente 91% no tamanho total do arquivo.

### Comparativo de Latência (Teste em Single-Core)
Para o teste, foi utilizado um processador **AMD Ryzen 5 5600** rodando em um ambiente WSL2, a execução foi limitada a um único núcleo de CPU via `taskset -c 0`:
* **Tempo Total H5:** 1.52s
* **Tempo Total TFLite:** 0.59s
* **Ganho de Eficiência:** O modelo TFLite foi aproximadamente **2.6x mais rápido**, demonstrando maior eficiência no uso da CPU.


**Resultado:** Sucesso na manutenção da inteligência do modelo com redução significativa da ocupação de memória e otimização para inferência até mesmo em CPU de baixa potência.

---

## 6. Como Executar o Projeto

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
    
