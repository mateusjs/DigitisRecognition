import numpy as np


def stdev(image):
    ans = np.zeros((4, 3))

    for linha in range(4):
        for coluna in range(3):
            ans[linha, coluna] = np.std(image[linha * 5:(linha * 5 + 5), coluna * 5:(coluna * 5 + 5)])

    return ans


def proportion(image):
    ans = np.zeros((4, 3))

    for linha in range(4):
        for coluna in range(3):
            ans[linha, coluna] = np.mean(image[linha * 5:(linha * 5 + 5), coluna * 5:(coluna * 5 + 5)] <= 127)

    return ans
