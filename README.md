# SuperResolutionApp

## Обзор
Этот проект, разработанный в рамках варианта 14, реализует супер-разрешение кадров видеопотока с камеры в реальном времени в зависимости от уровня размытости кадров. Приложение улучшает разрешение размытых кадров (Blur Score < 1000) с использованием предобученной модели FSRCNN, отображая оригинальный и улучшенный кадры рядом.

## Задание
Супер-разрешение кадров видеопотока в реальном времени в зависимости от уровня размытости кадров видеопотока.

## Используемые библиотеки
- **OpenCV Android SDK**: Используется для обработки изображений, включая преобразование YUV в RGB, определение размытости (через дисперсию Лапласиана) и интеграцию с моделью FSRCNN через модуль DNN.
- **Android CameraX**: Для доступа к камере и обработки видеопотока в реальном времени.
- **Jetpack Compose**: Для создания интерфейса, отображающего оригинальный и улучшенный кадры.

## Модель FSRCNN
Приложение использует модель **FSRCNN** (Fast Super-Resolution Convolutional Neural Network) для супер-разрешения. FSRCNN — это лёгкая нейросеть, разработанная для эффективного увеличения разрешения изображений, что делает её подходящей для мобильных устройств. Она обрабатывает Y-канал изображения (в цветовом пространстве YUV) и увеличивает его разрешение, а каналы U и V масштабируются соответствующим образом.

### Варианты модели
В папке `app/src/main/assets` находятся следующие варианты модели FSRCNN:
- **FSRCNN-small_x2.onnx** (по умолчанию): Лёгкая модель с масштабом x2 (увеличивает с 320x240 до 640x480). Выбрана как базовая версия за баланс скорости и качества.
- **FSRCNN_x2.onnx**: Полная модель с масштабом x2, обеспечивает лучшее качество, но требует больше вычислительных ресурсов.
- **FSRCNN_x4.onnx**: Модель с масштабом x4 (увеличивает с 320x240 до 1280x960), даёт наивысшее разрешение, но может вызывать лаги из-за высокой нагрузки.

Для лучшего разрешения можно выбрать модель x4, но придётся мириться с возможными лагами на слабых устройствах.

### Источник и конвертация моделей
Модели FSRCNN были взяты из репозитория Saafke/FSRCNN_Tensorflow. Изначально они были в формате TensorFlow (.pb) и конвертированы в ONNX (.onnx) с помощью команды:

```bash
python -m tf2onnx.convert \
  --graphdef FSRCNN-small_x2.pb \
  --output FSRCNN-small_x2.onnx \
  --inputs IteratorGetNext:0[1,-1,-1,1] \
  --outputs NCHW_output:0 \
  --opset 11
```

Для совместимости с модулем DNN в OpenCV из модели был удалён слой Unsqueeze с помощью следующего скрипта:
```py
import onnx
from onnx import helper
import numpy as np

# Загружаем модель
model_path = "lab6/FSRCNN_x4.onnx"
model = onnx.load(model_path)

# Получаем граф модели
graph = model.graph

# Ищем слой Unsqueeze
unsqueeze_node = None
for node in graph.node:
    if node.op_type == "Unsqueeze":
        unsqueeze_node = node
        break

if unsqueeze_node is None:
    print("Unsqueeze layer not found!")
    exit(1)

print("Found Unsqueeze node:", unsqueeze_node)

# Находим вход и выход Unsqueeze
unsqueeze_input = unsqueeze_node.input[0]
unsqueeze_output = unsqueeze_node.output[0]

# Удаляем Unsqueeze из графа
graph.node.remove(unsqueeze_node)

# Заменяем входы последующих узлов
for node in graph.node:
    for i in range(len(node.input)):
        if node.input[i] == unsqueeze_output:
            node.input[i] = unsqueeze_input

# Удаляем промежуточный тензор (выход Unsqueeze), если он больше не используется
for tensor in graph.value_info:
    if tensor.name == unsqueeze_output:
        graph.value_info.remove(tensor)
for output in graph.output:
    if output.name == unsqueeze_output:
        graph.output.remove(output)

# Проверяем модель
onnx.checker.check_model(model)

# Сохраняем изменённую модель
output_path = "lab6/FSRCNN_x4_no_unsqueeze.onnx"
onnx.save(model, output_path)

print(f"Model saved to {output_path}")
```

## Использование
1. Запустите приложение на устройстве Android.
2. Предоставьте разрешение на доступ к камере при запросе.
3. Приложение отобразит оригинальный видеопоток и улучшенный (если Blur Score < 1000).
4. Чтобы сменить модель FSRCNN, измените логику загрузки модели в MainActivity.kt и поместите нужную модель в `app/src/main/assets`.