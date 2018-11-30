from main import svm_builder, mlp_builder
from keras.utils import to_categorical


def sum():
    features_pro = np.load('feature_std.npy')
    label = np.load('label.npy')

    svmx_train, svmx_aux, svmy_train, svmy_aux = train_test_split(features_pro, label, test_size=0.5)
    svmx_validation, svmx_test, svmy_validation, svmy_test = train_test_split(svmx_aux, svmy_aux, test_size=0.5)

    features_pro = np.load('feature_prop.npy')
    label = np.argmax(np.load('label.npy'), axis=1)

    mlpx_train, mlpx_aux, mlpy_train, mlpy_aux = train_test_split(features_pro, label, test_size=0.5)
    mlpx_validation, mlpx_test, mlpy_validation, mlpy_test = train_test_split(mlpx_aux, mlpy_aux, test_size=0.5)

    svm = svm_builder(svmx_train, svmy_train)
    mlp = mlp_builder(mlpx_train, mlpy_train)

    out_svm = svm.predict(svmx_test)


def prod():
    pass


def max():
    pass
