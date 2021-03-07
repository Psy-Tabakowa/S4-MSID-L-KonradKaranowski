# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    hamming_matrix = np.sum((X.toarray()[:, np.newaxis, :] != X_train.toarray()), 2)
    return hamming_matrix


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    indexes = np.argsort(Dist, kind='mergesort')
    return y[indexes]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    n_samples, _ = y.shape
    y_neighbours = y[:, :k]
    n_classes = np.unique(y).size
    proba_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_classes):
        proba_matrix[:, i] = np.count_nonzero(y_neighbours == i, axis=1)
    return proba_matrix / k


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    n_samples, _ = p_y_x.shape
    # tak mogło być... ale musiałem wybrać ostatni największy index :(
    # pred = np.argmax(p_y_x, axis=1)
    pred = np.zeros(n_samples)
    for i, x in enumerate(p_y_x):
        pred[i] = np.argwhere(x == np.max(x))[-1]
    return np.sum(pred != y_true) / n_samples


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    errors = list()
    distance_matrix = hamming_distance(X_val, X_train)
    sorted_distance_matrix = sort_train_labels_knn(distance_matrix, y_train)
    for k in k_values:
        proba_matrix = p_y_x_knn(sorted_distance_matrix, k)
        error = classification_error(proba_matrix, y_val)
        errors.append(error)
    index_best_error = np.argmin(errors)
    return errors[index_best_error], k_values[index_best_error], errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    n_samples = y_train.size
    n_classes = np.unique(y_train).size
    y_priors = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        y_priors[i] = np.count_nonzero(y_train == i) / n_samples
    return y_priors


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    _, n_features = X_train.shape
    n_classes = np.unique(y_train).size
    p_x_y = np.zeros((n_classes, n_features), dtype=np.float32)
    for i in range(n_classes):
        idx = np.argwhere(y_train == i).reshape(-1)
        p_x_y[i] = (np.sum(X_train[idx], axis=0) + a - 1) / (idx.size + a + b - 2)
    return p_x_y


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    n_samples, _ = X.shape
    n_classes = p_y.size
    # to powinno być logarytmowane wszystko
    p_y_x = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, xi in enumerate(X.toarray()):
        p_y_x[i] = np.prod(
            p_x_1_y ** xi * (1 - p_x_1_y) ** (1 - xi),
            axis=1
        ) * p_y
    # zmień w tensor rzędu 2 bo numpy dostanie korby
    return p_y_x / p_y_x.sum(axis=1).reshape(-1, 1)


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Bayesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    errors = list()
    param_combinations = list()
    for a in a_values:
        error_inner = list()
        for b in b_values:
            p_y = estimate_a_priori_nb(y_train)
            p_x_y = estimate_p_x_y_nb(X_train, y_train, a, b)
            p_y_x = p_y_x_nb(p_y, p_x_y, X_val)
            error = classification_error(p_y_x, y_val)
            param_combinations.append((a, b))
            error_inner.append(error)
        errors.append(error_inner)
    index_best_error = np.argmin(errors)
    best_a, best_b = param_combinations[index_best_error]
    return np.min(errors), best_a, best_b, errors
