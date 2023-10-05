# Galym_Ayana
# Импортируем необходимые библиотеки
from sklearn.neighbors import KNeighborsClassifier

# Предположим, у нас есть набор данных с признаками фруктов
# Примеры признаков: вес (г), текстура (0 - гладкая, 1 - шероховатая), цвет (0 - красный, 1 - оранжевый)
X = [[140, 0, 0], [130, 0, 0], [150, 1, 1], [170, 1, 1]]
y = ['яблоко', 'яблоко', 'апельсин', 'апельсин']

# Создаем и обучаем модель KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X, y)

# Предсказываем класс нового фрукта
new_fruit = [[160, 1, 1]]  # Новый фрукт с весом 160 г, шероховатой текстурой и оранжевым цветом
predicted_class = knn_classifier.predict(new_fruit)

print(f"Предсказанный класс для нового фрукта: {predicted_class[0]}")
