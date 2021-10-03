
from math import sqrt
import time
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
arquivo_entrada_path = 'diabetes.csv'


time_start = time.time()

# https://www.kaggle.com/uciml/pima-indians-diabetes-database


def main():
    # open data in pandas
    data = pd.read_csv(arquivo_entrada_path)

    # correlation set from data
    correlation_set = data.corr()

    # columns variale names (first row minus last column)
    columns = list(data.head(0))
    columns.pop(-1)

    # predicted class = last column of first row
    target_variable = []
    target_variable.append(list(data.head(0))[-1])

    # x axys =  row values for each column
    x = data[columns]

    # y : goal class (outcome)
    y = data[target_variable]
    # 80% for training:
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8
    )
    # ordem de importância: correlacao do outcome em ordem crescente
    print('--- Ordem de importância---')
    correlation = correlation_set[target_variable]
    print(correlation.sort_values(target_variable, ascending=False))

    # n = 10
    print('--------------')
    print('-------Treinamento N = 10------')
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train.values.ravel())
    y_pred = rfc.predict(X_test)
    # precisao calculada por accuracy score

    train_score_RF = rfc.score(X_train, y_train)
    test_score_RF = rfc.score(X_test, y_test)
    print('-------Pontuação treinamento: ' + str(train_score_RF) + '-------')
    print("-------Pontuação teste: " + str(test_score_RF) + '-------')
    print('-------Precisão: ' + str(accuracy_score(y_test, y_pred)) + '-------')
    print('-------Matriz de confusão-------')
    print(confusion_matrix(y_test, y_pred))

    # n = 100
    print('--------------')
    print('-------Treinamento N = 100------')
    rfc1 = RandomForestClassifier(n_estimators=100)
    rfc1.fit(X_train, y_train.values.ravel())
    y_pred = rfc1.predict(X_test)
    train_score_RF = rfc1.score(X_train, y_train)
    test_score_RF = rfc1.score(X_test, y_test)

    print('-------Pontuação treinamento: ' + str(train_score_RF) + '-------')
    print("-------Pontuação teste: " + str(test_score_RF) + '-------')
    print('-------Precisão: ' + str(accuracy_score(y_test, y_pred)) + '-------')
    print('-------Matriz de confusão-------')
    print(confusion_matrix(y_test, y_pred))

    # N = raiz quadrada do tamanho da amostra
    sqrt_sample_size = round(math.sqrt(data.shape[0]))
    print('--------------')

    print('-------Treinamento N = ' + str(sqrt_sample_size) + '------')

    rfc2 = RandomForestClassifier(n_estimators=sqrt_sample_size)
    rfc2.fit(X_train, y_train.values.ravel())
    y_pred = rfc2.predict(X_test)
    train_score_RF = rfc2.score(X_train, y_train)
    test_score_RF = rfc2.score(X_test, y_test)

    print('-------Pontuação treinamento: ' + str(train_score_RF) + '-------')
    print("-------Pontuação teste: " + str(test_score_RF) + '-------')
    print('-------Precisão: ' + str(accuracy_score(y_test, y_pred)) + '-------')
    print('-------Matriz de confusão-------')
    print(confusion_matrix(y_test, y_pred))
    print('--------------')


if __name__ == '__main__':
    main()
    print('Execution Time:' + str(time.time() - time_start))
