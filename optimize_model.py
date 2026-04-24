import tensorflow as tf
import os


def main():

    model_path = 'model.h5'
    tflite_path = 'model.tflite'

    if not os.path.exists(model_path):
        print(f"Erro: O arquivo {model_path} não foi encontrado.")
        return

    print(f"Carregando o modelo Keras: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Aplicando quantização de pesos...")
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Gerado modelo otimizado: {tflite_path}")

    h5_size = os.path.getsize(model_path) / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024

    print("\n--- Relatório de Otimização de Espaço ---")
    print(f"Tamanho Original (.h5):     {h5_size:>8.2f} KB")
    print(f"Tamanho Otimizado (.tflite): {tflite_size:>8.2f} KB")
    print(
        f"Redução de Espaço:           {100 * (1 - tflite_size/h5_size):>8.1f}%")


if __name__ == "__main__":
    main()
