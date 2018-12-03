import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC



def mlp_builder(x_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(300, 250, 200), solver='sgd', tol=0, batch_size=500, max_iter=200,
                        shuffle=True,
                        learning_rate='adaptive', learning_rate_init=1e-3, power_t=.9)
    mlp.fit(x_train, y_train)
    return mlp


def svm_builder(x_train, y_train):
    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    return svm


def somatorio():
    features_std = np.load('feature_std.npy')
    features_pro = np.load('feature_prop.npy')
    label_std = np.load('label.npy')
    label_pro = np.argmax(np.load('label.npy'), axis=1)

    order = np.arange(features_pro.shape[0])
    np.random.shuffle(order)

    train_order = order[:int(features_pro.shape[0] / 2)]
    validation_order = order[int(features_pro.shape[0] / 2):int(features_pro.shape[0] * 3 / 4)]
    test_order = order[int(features_pro.shape[0] * 3 / 4):]

    svm = svm_builder(features_pro[train_order], label_pro[train_order])
    mlp = mlp_builder(features_std[train_order], label_std[train_order])

    pred_svm = svm.predict_proba(features_pro[test_order])
    pred_mlp = mlp.predict_proba(features_std[test_order])

    sum = np.argmax(np.add(pred_mlp, pred_svm), axis=1)
    prod = np.argmax(np.multiply(pred_mlp, pred_svm), axis=1)

    result = np.argmax(pred_svm, axis=1) != np.argmax(pred_mlp, axis=1)

    print(pred_svm[result])
    print(pred_mlp[result])

    # with open('combined.csv', 'a+') as file:
    #     file.write('%.4f %.4f\n' % (np.mean(sum == label_pro[test_order]), np.mean(prod == label_pro[test_order])))


def prod():
    pass


def max():
    pass
