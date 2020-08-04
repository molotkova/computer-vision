# Computer Vision projects
Мини-проекты во время стажировки в компании Цифра и изучения библиотеки OpenCV.
## Object tracking without ML
### Детектирование объекта
Для начала нужно было научиться детектировать объект, перебрала следующие варианты:

1) Фукнция cv2.matchTemplate() сравнивает шаблон с участками картинки, можно задавать порог (чувствительность), с которого объекты считаются одинаковыми. Хорошо работает для поиска идентичных шаблону участков на картинке, но при малом изменении искомого объекта (из-за угла съёмки или масштаба, например) алгоритм поиска работает плохо. Подробнее с вариацией методов функции на примерах смотреть [здесь](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html).

2) Иногда при поиске определённого объекта можно цепляться за формы. Например, на картинке ниже можно научиться находить круг на дне чашки, но у меня возникли проблемы с устойчивостью. На видео во время движения кружки меняется угол освещения, и круг на донышке находится с сильными перебоями.

![example](https://github.com/molotkova/computer-vision/blob/master/git_rnd.jpg)

3) Поиск по цвету. Этот метод показал себя лучше всего в смысле устойчивости, в итоге именно он был реализован. Алгоритм: чтение изображения и перегон в палитру HSV (эффективна для работы с цветом) -> нахождение участков, содержащих цвет в пределах установленных границ (в примере на скринкасте с синей чашкой использую границы (110, 0, 0) - (120, 255, 255)), выделение контуров -> выбор самого длинного контура - это и есть искомый объект, выделяю его в красный бокс.

### Трекинг и измерение расстояний

С помощью cv2.getPerspectiveTransform() делаю преобразование координатной сетки на выделенном участке с учётом перспективы для корректного измерения расстояний (уменьшаю искажения, связанные с разной удалённостью точек от камеры). Также преобразую координаты центра бокса.

Дальше калибрую камеру по нижней грани доски на заднем плане. Калибровка - это "правило" перевода длины из числа пикселей в см. Калиброваться можно по любому предмету известной длины, хорошо видном на видео, если камера зафиксирована, достаточно это сделать один раз.

\*длины на видео - это расстояния от центра красного бокса до границ верхней, правой, нижней и левой граней жёлтой области

**На перспективу:** несложно научиться детектировать сразу несколько чашек *разных* цветов и измерять расстояния между ними.

**Ограничения:** посторонние объекты того же цвета, что и искомый объект будут провоцировать сбои в работе алгоритма. Программа будет периодически "отвлекаться" на них.
### Источники, которые очень помогли
* [Детекция объектов по цвету](https://towardsdatascience.com/real-time-object-detection-without-machine-learning-5139b399ee7d)

* Динамичное измерение расстояний [1](https://towardsdatascience.com/monitoring-social-distancing-using-ai-c5b81da44c9f?gi=50a5df69723b) и [2](https://towardsdatascience.com/a-social-distancing-detector-using-a-tensorflow-object-detection-model-python-and-opencv-4450a431238)

* Работа с цветом [1](https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/) и [2](https://robotclass.ru/tutorials/opencv-color-range-filter/)

* [Color picker](https://imagecolorpicker.com/)

* [Палитра HSV в OpenCV](https://answers.opencv.org/question/184711/select-hsv-hue-from-30-to-30-in-python/)
