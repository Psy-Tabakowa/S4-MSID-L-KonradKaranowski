
# Lista 1 – Regresja liniowa

### Do zadań 1 i 2 zdefiniujmy sobie kilka rzeczy.

Mamy dany wektor cech:
<br>
![image](https://cdn.mathpix.com/snip/images/kCQLn_flAx1heMERK_Bk7FeNdqubDnIINTtjCKqSxuk.original.fullsize.png)

Oraz wektor wyników:
<br>
![image](https://cdn.mathpix.com/snip/images/30U_X2VPcedlcGjbnDd4SecM7uLzbdFDQZbp5GAgf0U.original.fullsize.png)

Tworzymy design matrix X o wymiarach n x m z naszego wektora x:
<br>
![image](https://cdn.mathpix.com/snip/images/HvN_jNxIPPOvyympIVnOFrd0vLXIwOfKaKoshnKuT3w.original.fullsize.png)

Dobierzmy teraz wektor wag:
<br>
![image](https://cdn.mathpix.com/snip/images/f3eGW-ultwEloc0q8TFcsPdM-jVk59hnKqiQQdOZzpQ.original.fullsize.png)

Teraz aby policzyć przewidziane wyniki wystarczy pomnożyć macierz X oraz w:
<br>
![image](https://cdn.mathpix.com/snip/images/FkPJzN0QGtrjM1ASfbqlrqDEv0iTwc4t0T3rE60-AIw.original.fullsize.png)

Zapiszmy to ładniej i trzymajmy się tego:
<br>
![image](https://cdn.mathpix.com/snip/images/xqNV8dzOY9W_cGK77H1MoarACe38ddy5B9Bm7oqaIQs.original.fullsize.png)

Będzimy też korzystać z następujących wzorów na pochodne:
<br>
![image](https://cdn.mathpix.com/snip/images/PUzxY1boz_Ht1iUMg3YHJJQwWd6SW1yv4auC0RHWAy8.original.fullsize.png)

### 1. Proszę wyznaczyć rozwiązanie liniowego zadania najmniejszych kwadratów.
Zdefiniujmy sobie naszą funkcję kosztu:
<br>
![image](https://cdn.mathpix.com/snip/images/ynXHi03uwvI1ktNlERvF_2rgVE-x9H_X8Mg6ipBm33g.original.fullsize.png)

Możemy to zapisać sprytniej:
<br>
![image](https://cdn.mathpix.com/snip/images/ibgODlxZ8VusLXgZZ6VOvaE9qYW2MUsw0sy8rZ8ZEIo.original.fullsize.png)

Rozpiszmy to sobie bardziej:
<br>
![image](https://cdn.mathpix.com/snip/images/JUeCicZ0zxGnykHZ7qn6dLtCNNeTstUXSdUKBUNs0R8.original.fullsize.png)

Liczymy sobie pochodną funkcji kosztu po wektorze wag i przyrównujemy ją do zera:
<br>
![image](https://cdn.mathpix.com/snip/images/UDhV4sKOJDzMnt0ANkqLdJJkkvg0b1SI0pof04qOgKs.original.fullsize.png)

Otrzymujemy następujący wzór:
<br>
![image](https://cdn.mathpix.com/snip/images/lt1gs7ZT0Ig4TqhAR-flIIm0CwBw-CnLGaJP6Et98zM.original.fullsize.png)

Przekształcamy równanie do postaci:
<br>
![image](https://cdn.mathpix.com/snip/images/Bj-l1j08x2BSCPQLy_cCWWq6bs-Z3JXiEqrxM88pRMs.original.fullsize.png)

Otrzymujemy ostateczny wzór:
<br>
![image](https://cdn.mathpix.com/snip/images/2m0H_Y-u5KLunwou9czqoON-vDnNoAXfPZWbkWg2Q2k.original.fullsize.png)

### 2. Proszę wyznaczyć rozwiązanie liniowego zadania najmniejszych kwadratów z regularyzacją L2.
Zdefiniujmy sobie naszą funkcję kosztu:
<br>
![image](https://cdn.mathpix.com/snip/images/bdPlYM6YyYrUDvFikMbaCJp-65VdW_SISLEUBW8H-7A.original.fullsize.png)

Dokonujemy analogicznych przekształceń co w poprzednim przykładzie:
<br>
![image](https://cdn.mathpix.com/snip/images/aENFncXLFe1M_NxYOSPJS7MtgLjxSqpoQJmVHhgiwm4.original.fullsize.png)

Liczymy pochodną:
<br>
![image](https://cdn.mathpix.com/snip/images/liZpidIPA5sp4faHvexAr_Ud9ynSA6MWCx2khItYFT0.original.fullsize.png)

Przekształcamy:
<br>
![image](https://cdn.mathpix.com/snip/images/8V1MG0yWz4-cc9tvgYqZw8K_UQDOlLmpGiWateNiNUc.original.fullsize.png)

Otrzymujemy nasze piękne rozwiązanie równania:
<br>
![image](https://cdn.mathpix.com/snip/images/hsuTjexnC3qpKheqX1I6qocPlrAQzg2kXuafuHyE-9w.original.fullsize.png)


### 3. Co to jest overfitting? Wskazać na przykładzie dopasowania wielomianu.
Overfitting polega na nadmiernym dopasowaniu do danych (mówimy, że model posiada dużą wariancję). W efekcie model dobrze dopasowuje się do danych treningowych, ale słabo radzi sobie z generalizacją (czyli słabo sprawdza się w realnym zastosowaniu).

### 4. Co to jestunderfitting? Wskazać na przykładzie dopasowania wielomianu.
Underfitting polega na niedostatecznym dopasowaniu do danych (mówimy, że model posiada duże obciążenie). W efekcie model nie dopasowuje się dobrze do danych treningowych i nie radzi sobie z generalizacją (czyli jest beznadziejny).

Tutaj przyład z [medium](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76):
![image](https://miro.medium.com/max/1125/1*_7OPgojau8hkiPUiHoGK_w.png)
### 5. Co to jest ciąg treningowy, walidacyjny, testowy? Jakie jest ich znaczenie.
Dane dzielimy najczęściej na trzy zbiory:

- Treningowy – na nim trenujemy nasz model
- Walidacyjny – na nim badamy skuteczność naszego modelu w trakcie trenowania
- Testowy – na nim sprawdzamy ostatecznie nasz model

Podział na te trzy ciągi jest bardzo istotny, ponieważ pozwala nam oceniać skuteczność naszego modelu (przykładowo, jeżeli model ma 90% skuteczności dla danych treningowych, ale 40% dla danych testowych, to wiemy, że występuje overfitting). Dzięki danym walidacyjnym możemy patrzeć na bieżąco w trakcie trenowania czy nasz algorytm rzeczywiście się uczy (to przydaje się szczególnie przy trenowaniu głębokich sieci neuronowych o wielu milionach parametrów). Często stosowaną regułą przy dzieleniu danych na ciągi jest podzielenie na dane treningowe i testowe w proporcjach 80/20, a następnie wydzielenie 20% danych walidacyjnych ze zbioru treningowego.

### 6. Co  to  jest  selekcja  modelu?  W  jaki  sposób  się  ją  wykonuje?  Czy  miara  oceniająca  model może być inna od kryterium uczenia?
Selekcja modelu polega na stworzeniu kilku modeli z różnymi hiperparametrami i wybraniu najlepszego. Kryterium może być inne np. przy klasyfikacji zamiast accuracy użyjemy F1 score.

### 7. Które z podejść do selekcji modelu jest prostsze do zastosowania w praktyce i dlaczego?
Ogólnie protsze jest zastosowanie tego samego kryterium, ponieważ nie musimy wykonywać dodatkowych obliczeń, ale w praktyce: to zależy.

### 8. Kiedy liniowe zadanie najmniejszych kwadratów ma jednoznaczne rozwiązanie, a kiedy istnieje wiele rozwiązań? Jak jest w przypadku zadania najmniejszych kwadratów z regularyzacją?
Rozwiązanie ma jednoznaczne rozwiązanie, kiedy rank(X) = M (macierz X.T * X nie jest osobliwa).
W przypadku równania z regularyzacją, zawsze istnieje jednoznaczne rozwiązanie.

### 9. Zapisać wektor cech φ dla wielomianu M-tego rzędu.
![image](https://cdn.mathpix.com/snip/images/HvN_jNxIPPOvyympIVnOFrd0vLXIwOfKaKoshnKuT3w.original.fullsize.png)

### 10. Co to jest parametr λ? Jak jego wartość wpływa na rozwiązanie?
Lambda jest tzw. współczynnikiem regularyzacji. Regularyzację stosujemy, aby nasz model się nie overfittował. W naszym przypadku polega ona na penalizowaniu zbyt wysokich wag.