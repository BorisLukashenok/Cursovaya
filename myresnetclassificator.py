# подключаем библиотеку компьютерного зрения
import cv2
import torch
import torchvision.transforms as tt

# функция определения лиц


def highlightFace(net, frame, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn = frame.copy()
    # высота и ширина кадра
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections = net.forward()
    # переменная для рамок вокруг лица
    faceBoxes = []
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence = detections[0, 0, i, 2]
        # если результат превышает порог срабатывания — это лицо
        if confidence > conf_threshold:
            # формируем координаты рамки
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1, y1, x2, y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    # возвращаем кадр с рамками
    return frameOpencvDnn, faceBoxes


def handClassificator(net, img):
    CLASSNAME = ['palm',
                 'l',
                 'fist',
                 'fist_moved',
                 'thumb',
                 'index',
                 'ok',
                 'palm_moved',
                 'c',
                 'down']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = cv2.merge((image, image, image))
    cv2.imshow('ee', image)
    image = torch.FloatTensor(image)
    
    output = net(image.permute(2, 0, 1).unsqueeze(0).cuda())
    return CLASSNAME[output[0].argmax()]
   


# загружаем веса для распознавания лиц
faceProto = "Model/opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel = "Model/opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet = cv2.dnn.readNet(model=faceModel, config=faceProto)
resnet = torch.load('./Model/resnet.pth')
resnet.eval()
# получаем видео с камеры
video = cv2.VideoCapture(0)
# пока не нажата любая клавиша — выполняем цикл
while cv2.waitKey(1) < 0:
    # получаем очередной кадр с камеры
    hasFrame, frame = video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break
    # распознаём лица в кадре
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    classes = handClassificator(resnet, frame)
    print(classes)
    # если лиц нет
    # if not faceBoxes:
    #     # выводим в консоли, что лицо не найдено
    #     print("Лица не распознаны")
    # выводим картинку с камеры
    cv2.imshow("Face detection", resultImg)
