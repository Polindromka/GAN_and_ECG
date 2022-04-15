# GAN_and_ECG
## Запуск программы:
```
python main.py 'params/main.json'
```
## Запуск предобработки данных:
```
python preload.py 'params/preload.json'
```
## Запуск обучения GAN:
```
python generator_training.py 'params/generator_training.json'
```
## Запуск генерации изображений:
```
python generator.py 'params/generator.json'
```
## Запуск обучения классификатора:
```
python classifier_training.py 'params/classifier_training.json'
```
## Запуск тестирования классификатора:
```
python classifier.py 'params/classifier.json'
```
