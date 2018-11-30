import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def svm():
    features_pro = np.load('feature_prop.npy')
    label = np.argmax(np.load('label.npy'), axis=1)

    x_train, x_aux, y_train, y_aux = train_test_split(features_pro, label, test_size=0.5)
    x_validation, x_test, y_validation, y_test = train_test_split(x_aux, y_aux, test_size=0.5)

    svm = LinearSVC()
    svm.fit(x_train, y_train)

    with open('svm.csv', 'a+') as file:
        file.write('%.4f %.4f %.4f\n' % (
        svm.score(x_train, y_train), svm.score(x_test, y_test), svm.score(x_validation, y_validation)))


def mlp():
    features_pro = np.load('feature_std.npy')
    label = np.load('label.npy')

    x_train, x_aux, y_train, y_aux = train_test_split(features_pro, label, test_size=0.5)
    x_validation, x_test, y_validation, y_test = train_test_split(x_aux, y_aux, test_size=0.5)

    mlp = MLPClassifier(hidden_layer_sizes=(300, 250, 200), solver='sgd', tol=0, batch_size=500, max_iter=200,
                        shuffle=True,
                        learning_rate='adaptive', learning_rate_init=1e-3, power_t=.9)

    mlp.fit(x_train, y_train)

    with open('mlp.csv', 'a+') as file:
        file.write('%.4f %.4f %.4f\n' % (
        mlp.score(x_train, y_train), mlp.score(x_test, y_test), mlp.score(x_validation, y_validation)))


for _ in range(10):
    print('iteration:', _)
    # svm()
    mlp()
