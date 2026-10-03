# Spaceship Titanic

Решение в: task.ipynb

## Описание решения

1. Загрузил данные и изучил признаки, их типы и пропущенные значения.
2. Для подготовки данных создал собственные трансформеры, совместимые со sklearn Pipeline:
    - один трансформер разделял PassengerId на GroupId и InGroupId
    - второй разделял Cabin на CabinDeck, CabinNum и CabinSide
3. Собрал единый Pipeline предобработки данных с помощью ColumnTransformer:
    - числовые признаки заполнял через IterativeImputer и масштабировал MinMaxScaler
    - бинарные признаки преобразовывал в float и заполнял пропуски наиболее частым значением
    - категориальные признаки заполнял наиболее частым значением и преобразовывал через OneHotEncoder
4. Проверил несколько моделей с помощью cross-validation и сравнил их по accuracy: LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier.
5. Для KNeighborsClassifier и RandomForestClassifier использовал GridSearchCV и перебирал различные значения гиперпараметров.
6. Затем протестировал LGBMClassifier и с помощью GridSearchCV перебрал learning_rate, n_estimators и num_leaves.
7. После первого поиска сузил диапазон гиперпараметров LGBMClassifier вокруг лучших найденных значений и повторно запустил GridSearchCV.
8. В результате получил лучшую комбинацию параметров
9. Обучил с ней итоговую модель на всех обучающих данных, используя тот же Pipeline предобработки.
10. Получил предсказания для тестовой выборки и сохранил их в submission.csv.

Public Score на kaggle:
Accuracy = 0.79798