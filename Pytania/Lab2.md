
# Lista 2 – KNN i Naiwny Klasyfikator Bayesa


### 1.  Proszę wyznaczyć estymator największej wiarygodności dla rozkładu wielopunktowego.
Parametry theta muszą spełniać następującą właściwość:
<br>
![](https://cdn.mathpix.com/snip/images/6PK4NKJztYKbWMH9h-cKMPwA3qcGODP_bPJ-GV4-MnQ.original.fullsize.png)

Funkcja wiarygodności ma postać:
<br>
![](https://cdn.mathpix.com/snip/images/ms5BhYehRcFLpmwSslC0DeEr_PGftaNuUj_4u6oofjE.original.fullsize.png)

Logarytmujemy stronami:
<br>
![](https://cdn.mathpix.com/snip/images/um68ipM9zYMAA-26DlOT6gbNO2_uuv73R6xQe7FpBQI.original.fullsize.png)

Liczymy gradient naszej logarytmowanej funkcji wiarygodności:
<br>
![](https://cdn.mathpix.com/snip/images/gCudG_mw1HQtYiXcTMTNBKt5X0Q5KZGXb2WbBLy08sM.original.fullsize.png)

Liczymy gradient parametrów:
<br>
![](https://cdn.mathpix.com/snip/images/502G8pHdklWF0k0bQLOzvaky6BWxy3SdAzqI_dgAKhM.original.fullsize.png)

Zauważamy, że wektory są proporcjonalne i korzystamy z metody mnożników Lagrange'a:
<br>
![](https://cdn.mathpix.com/snip/images/TYu8Tg3gEwGFAmWrVzKFY27U49KCUTpfJUxXppWB5gY.original.fullsize.png)

Rozwiązujemy układ równań:
<br>
![](https://cdn.mathpix.com/snip/images/cLrhtNfLEXas63UKbrTZwX2WQLRKCQOwxoSyVfnirKA.original.fullsize.png)

Otrzymujemy wynik:
<br>
![](https://cdn.mathpix.com/snip/images/IU58c7BK3X5ofrDJhP6oam8zeHrdB_CAm0alBXm3ga0.original.fullsize.png)



### 2.  Proszę wyznaczyć estymator największej wiarygodności dla rozkładu dwupunktowego.
Szukamy takiej wartości θ, dla której łączne prawdopodobieństwo p(X|θ) jest największe, formalnie możemy zapisać to tak:
<br>
![](https://cdn.mathpix.com/snip/images/LSH22umxMxH9sUsa0tUlJxN7Cej8LMxKC9zwTSPVQFo.original.fullsize.png)

Gdzie L, to funkcja wiarygodności.
Załóżmy zbiór naszych danych:
<br>
![](https://cdn.mathpix.com/snip/images/UiwJ3alsXdW-JiuYtURVKcNMrBZpf9i9NR94s0E4xyw.original.fullsize.png)

Prawdopodobieństwo dla zmiennej danej rozkładem Bernoulliego jest równe:
<br>
![](https://cdn.mathpix.com/snip/images/KQkbzyg_FIGdK_eDSkI8zc9YNP9TTgN5x_9F7Gu6UOk.original.fullsize.png)

Funkcja wiarygodności ma postać:
<br>
![](https://cdn.mathpix.com/snip/images/eQmrmE9KVDB1fPeF_XcYPkMnEKoVziMLIqipx_EqA68.original.fullsize.png)

Logarytmujemy stronami i przekształcamy:
<br>
![](https://cdn.mathpix.com/snip/images/xMKJbOkm5g72JTsorH_40xTxNZJBLeCGG9EgbRZUJro.original.fullsize.png)

Liczymy pochodną z tej funkcji:
<br>
![](https://cdn.mathpix.com/snip/images/wU41HTzPxyDnHpSZN8gxnw7Hcny3ARIiX1qBx05KMlY.original.fullsize.png)

Przyrównujemy do zera i przekształcamy:
<br>
![image](https://cdn.mathpix.com/snip/images/BVaKpHWkfIWfYUCVqW4VXU31PqhpbyugSHGDgIk3tYE.original.fullsize.png)


Otrzymujemy ostateczną postać:
<br>
![](https://cdn.mathpix.com/snip/images/878FBIMtZsJeFU7cuBEIgva8wIEK9-yrxs7GtKslYyI.original.fullsize.png)


### 3.  Proszę wyznaczyć estymator maksymalnego a posteriori dla rozkładu dwupunktowego.
Szukamy takiego θ, dla którego prawdopodobieństwo p(θ|X) jest największe. Formalnie możemy zapisać to tak:
<br>


Ponownie definiujemy nasz zbiór danych:
<br>
![](https://cdn.mathpix.com/snip/images/UiwJ3alsXdW-JiuYtURVKcNMrBZpf9i9NR94s0E4xyw.original.fullsize.png)

Prawdopodobieństwo dla zmiennej danej rozkładem Bernoulliego jest równe:
<br>
![](https://cdn.mathpix.com/snip/images/KQkbzyg_FIGdK_eDSkI8zc9YNP9TTgN5x_9F7Gu6UOk.original.fullsize.png)

Zakładamy pewien rozkład prawdopodobieństwa a priori p(θ).
Interesuje nas znalezienie takiego θ, dla którego prawdopodobieństwo a posteriori p(θ|X) jest najwyższe. Korzystając z reguły Bayesa:
<br>
![](https://cdn.mathpix.com/snip/images/fGz89rgSRHyvGQSv4F9SPGDmWlRt23ebsS8VNMw9toA.original.fullsize.png)

Jako, że p(X) jest stałe, p(θ|X) ma wartość najwyższą, gdy p(θ)p(X|θ) ma wartość najwyższą. Teraz nasz estymator MAP ma postać:
<br>
![](https://cdn.mathpix.com/snip/images/PpcSdIADIO_cTFUsQE9DeOAb7MD4miCqzjo9Ewskx38.original.fullsize.png)


Zakładamy, że rozkład p(θ) jest rozkładem Beta (bo tak jest w zadaniu):
<br>
![](https://cdn.mathpix.com/snip/images/AC-O1SnTipbLzYB3XyLYAE7QA-XySTrF3XZJ7Rexsx0.original.fullsize.png)

Funkcja wiarygodności ma postać:
<br>
![](https://cdn.mathpix.com/snip/images/C3Y5Ys5CsFH5CKKK4w48uqOnqnhPSRLoJly19AWJtRQ.original.fullsize.png)
<br>
![](https://cdn.mathpix.com/snip/images/dYv6anLRKxBF5BSup79r_OvpzeUUUx8j9gSD1oqAN3w.original.fullsize.png)

Logarytmujemy stronami:
<br>
![](https://cdn.mathpix.com/snip/images/lHnzp6ezzfcwxMUNcqw1Nk3KKR_iEgDepBbAIixmv14.original.fullsize.png)


Liczymy pochodną:
<br>
![](https://cdn.mathpix.com/snip/images/bQEC6ZOEyfvknpbnNUYgg_yLiXsxDPieiOd0ctLDA7Q.original.fullsize.png)

Przyrównujemy ją do zera i przekształcamy:
<br>
![](https://cdn.mathpix.com/snip/images/rNYGsl4rrmMcS57V_Xi7esjOAXd0zb6yJFf8hqeh7Zg.original.fullsize.png)

Otrzymujemy następujący wynik:
<br>
![](https://cdn.mathpix.com/snip/images/euCJmQ0MOiPZhO7UOxeJCmikXAi_qVGEgqml0Lnc1yk.original.fullsize.png)

### 4.  Dlaczego stosujemy założenie o niezależności cech określających wystąpienie słowa w dokumencie? Jaka jest korzyść z takiego podejścia, a jaka jest strata?
Stosujemy założenie o niezależności cech, aby model był w ogóle wyuczalny w skończonym czasie.
Tracimy w ten sposób możliwość modelowania zależności między cechami, ale zyskujemy prosty i w miarę skuteczny model.


### 5.  Jaka jest interpretacja parametrów θ? Ile jest takich parametrów dla D cech i K klas?
θ to prawdopodobiestwa warunkowe, że dana cecha przyjmuje wartość 1 (występuje) pod warunkiem, że próbka pochodzi z klasy k.
Jest ich D x K


### 6.  Jaka jest interpretacja parametrów π? Ile jest takich parametrów dla D cech i K klas?
Parametry te to prawdopodobieństwa a priori wystąpienia danej klasy, jest ich K.


### 7.  Jaka jest interpretacja hiperparametru k? Za co odpowiada? Jaka jest jego interpretacja geometryczna? Jak jego wartość wpływa na rozwiązanie?
K to ilość najbliższych sąsiadów, których bierzemy pod uwagę przy klasyfikacji nowej próbki.
Im większe jest k, tym więcej sąsiadów bierzmy pod uwagę. 
Im większe k, tym bardziej skomplikowany jest model.


### 8.  W jaki sposób wyznaczamy sąsiedztwo w modelu k-NN?
Sąsiedztwo stanowią inne punkty, które są położone w jakieś odległości od siebie. Wyznaczamy je licząc odległości między punktami 
(np. za pomocą metryki Minkowskiego). Następnie wybieramy k najbliższych punktów względem danego punktu. 

### 9.  Czy model k-NN jest modelem generującym, czy dyskryminującym? Czy jest to model parametryczny, czy nieparametryczny?
KNN jest modelem: 
* nieparametrycznym, ponieważ zapamiętujemy cały zbiór danych treningowych (tzw. model leniwego ucznia)
* dyskryminującym, ponieważ wyznaczamy prawdopodobieństwo warunkowe p(y|x) bez modelowania całego rozkładu prawdopodobieństwa

### 10.  Czy model Naive Bayes jest modelem generującym, czy dyskryminującym? Czy jest to model parametryczny, czy nieparametryczny?
Naive Bayes jest modelem: 
* parametrycznym, ponieważ w procesie uczenia wyznaczamy parametry θ oraz π
* generycznym, ponieważ modelujemy cały rozkład prawdopodobieństwa p(x|y) i p(y)
