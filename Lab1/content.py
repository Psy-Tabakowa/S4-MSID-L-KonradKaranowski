# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    pred = polynomial(x, w)
    return ((y - pred) ** 2).mean()


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    return np.vander(x_train.reshape(-1), M + 1, increasing=True)


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    vander_matrix = design_matrix(x_train, M)
    weights = np.linalg.inv(vander_matrix.T @ vander_matrix) @ vander_matrix.T @ y_train
    error = mean_squared_error(x_train, y_train, weights)
    return weights, error


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    vander_matrix = design_matrix(x_train, M)
    identity = np.eye(M + 1)
    weights = np.linalg.inv(
        vander_matrix.T @ vander_matrix + identity * regularization_lambda
    ) @ vander_matrix.T @ y_train
    error = mean_squared_error(x_train, y_train, weights)
    return weights, error


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    models = list()
    for m in M_values:
        weights, train_loss = least_squares(x_train, y_train, m)
        val_loss = mean_squared_error(x_val, y_val, weights)
        models.append(
            (
                weights,
                train_loss,
                val_loss
            )
        )
    sorted_models = sorted(models, key=lambda x: x[2])
    return sorted_models[0]


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    models = list()
    for regularization_lambda in lambda_values:
        weights, train_loss = regularized_least_squares(x_train, y_train, M, regularization_lambda)
        val_loss = mean_squared_error(x_val, y_val, weights)
        models.append(
            (
                weights,
                train_loss,
                val_loss,
                regularization_lambda
            )
        )
    sorted_models = sorted(models, key=lambda x: x[2])
    return sorted_models[0]
