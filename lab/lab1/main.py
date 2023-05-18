
import numpy as np
import pandas as pd
from sklearn.datasets import *
import matplotlib.pyplot as plt
import seaborn as sns

def task_1():
    print("\n" * 100)
    print("Преобразование наборов данных Scikit-learn в Pandas Dataframe")
    iris_data = load_wine() # Загрузка
    print("Тип Датасета")
    print(type(iris_data)) #тип

    print("Ключи датасета")
    for x in iris_data:
        print(x)
    print("Колонки датасета")
    for item in iris_data['feature_names']:
        print(item)
    print("По итогу имеем датасет")
    data1 = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                         columns=iris_data['feature_names'] + ['target'])
    print(data1.head())

def get_data():
    iris_data = load_wine()
    data1 = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                         columns=iris_data['feature_names'] + ['target'])
    return data1
def setSettings():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
def task_2():
    data = get_data()

    print('Данные представляют собой результаты химического анализа вин, выращенных в одном и том же регионе Италии \nтремя разными производителями. Проведено тринадцать различных измерений \nразличных компонентов, содержащихся в трех винах.')
    print('Описание полей')
    print('Alcohol:алкоголь в напитке \nMalic Acid: Яблочная кислота\nAsh:Зола\nAlcalinity of Ash:Щелочность золы:\nMagnesium:Магний\nTotal Phenols:Общие фенолы\nFlavanoids:Флаваноиды\nNonflavanoid Phenols:Нефлаваноидные фенолы\nProanthocyanins:Проантоцианы')
    print('Colour Intensity:Интенсивность цвета\nHue:оттенок\nOD280/OD315 of diluted wines:OD280/OD315 разбавленных вин:\nProline:Пролин')

    print("Первые строки датасета")
    print(data.head())
    print("Размеры датасета")
    rows, columns = data.shape
    print(f'Строк: {rows}\nКолонок: {columns}')
    print(f'Колонки: {"; ".join([item for item in data.columns])}')

    print(f'Типы колонок\n{data.dtypes}')
    print('Проверка пустых значений')
    missingData = [print(f'{col}: {data[data[col].isnull()].shape[0]}') for col in data.columns]

    print('Рассмотрим основные статистические характеристики набора данных')
    print(data.describe())


    print('Определим уникальные значения целевого признака')
    print(data['target'].unique())

    print('Визуально исследуем датасет')
    print('Построим диаграмму рассеивания')


    sns.scatterplot(x='ash', y='target', data=data)
    plt.show()

    print('Рассмотри кол-во алкоголя в рассматриваемых напитках')
    sns.distplot(data['alcohol'])
    plt.show()

    print('Построим "Ящик с усами" для анализа кол-ва Пролина в различных примерах')
    sns.boxplot(x='target', y='proline', data=data)
    plt.show()

    print('Построим и проанализируем матрицу корреляции признаков')
    print(data.corr())

    print('Для анализа построим тепловую карту, вычисляя коэффициент корреляциии тремя различными методами.')



    plt.rcParams['figure.figsize'] = [10, 8]
    f1 = sns.heatmap(data.corr(method='spearman'), annot=True, fmt='.1f',)
    f1.set_title('spearman', fontdict={'fontsize':12}, pad=12)
    plt.show()
    f2 = sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.1f')
    f2.set_title('pearson', fontdict={'fontsize':12}, pad=12)
    plt.show()
    f3 = sns.heatmap(data.corr(method='kendall'), annot=True, fmt='.1f')
    f3.set_title('kendall', fontdict={'fontsize': 12}, pad=12)
    plt.show()


if __name__ == '__main__':

    flag = True
    setSettings()
    while(flag):
        print('Выберите номер операции \n1 - задание 1 \n2 - задание 2\n3 - выход')
        try:
            type_of_operation = int(input())
        except:
            print("Неверный ввод повторите снова")

        if type_of_operation == 1:
            task_1()
        elif type_of_operation == 2:
            task_2()
        elif type_of_operation == 3:
            flag = False
        else:
            pass







