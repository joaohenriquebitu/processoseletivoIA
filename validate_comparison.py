import numpy as np
import tensorflow as tf
import time

# Este script foi desenvolvido exclusivamente para validar e garantir a integridade do modelo após a otimização, comparando o modelo original com o otimizado, não fazendo parte do processo de treino e otimização.


def main():

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_norm = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model_h5 = tf.keras.models.load_model("model.h5")

    start_time = time.time()
    loss_h5, acc_h5 = model_h5.evaluate(x_test_norm, y_test, verbose=0)
    duration_h5 = time.time() - start_time

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    hits_tflite = 0
    start_time = time.time()

    for i in range(len(x_test_norm)):
        input_data = np.expand_dims(x_test_norm[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output_data) == y_test[i]:
            hits_tflite += 1

    acc_tflite = hits_tflite / len(y_test)
    duration_tflite = time.time() - start_time

    print("       RESULTADOS FINAIS")
    print("================================")
    print(f"Acurácia H5:     {acc_h5:.4f}")
    print(f"Acurácia TFLite: {acc_tflite:.4f}")
    print(f"Diferença:       {abs(acc_h5 - acc_tflite):.6f}")
    print("--------------------------------")
    print(f"Tempo Total H5:     {duration_h5:.2f}s")
    print(f"Tempo Total TFLite: {duration_tflite:.2f}s")
    print("================================")


if __name__ == "__main__":
    main()
