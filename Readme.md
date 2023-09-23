# Курсовая работа

## Задача 

>Нужно написать приложение, которое будет считывать и выводить кадры с веб-камеры.  
>В процессе считывания определять что перед камерой находится человек, задетектировав его лицо на кадре.  
>Человек показывает жесты руками, а алгоритм должен считать их и классифицировать.  

## Создание и обучение модели

[Ноутбук с обучением:](Cursovaya_PyTorch.ipynb)

## Приложение работающее с моделью

[myresnetclassificator.py](myresnetclassificator.py)

Не работает. Сенсора Leap Motion  у меня нет. Не удалось преобразовать картинку с обычной вебкамеры похожей на сферическое ИК-изображение

## Распознование жестов на основе mediapipe
[mediapipeclassificator.py](mediapipeclassificator.py)