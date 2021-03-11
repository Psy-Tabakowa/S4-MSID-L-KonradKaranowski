
# Lista 2 – KNN i Naiwny Klasyfikator Bayesa


### 1.  Proszę wyznaczyć estymator największej wiarygodności dla rozkładu wielopunktowego.


### 2.  Proszę wyznaczyć estymator największej wiarygodności dla rozkładu dwupunktowego.


### 3.  Proszę wyznaczyć estymator maksymalnego a posteriori dla rozkładu dwupunktowego.


### 4.  Dlaczego stosujemy założenie o niezależności cech określających wystąpienie słowa w dokumencie? Jaka jest korzyść z takiego podejścia, a jaka jest strata?
Stosujemy założenie o niezależności cech, aby model był w ogóle wyuczalny w skończonym czasie.
Tracimy w ten sposób możliwość modelowania zależności między cechami, ale zyskujemy prosty i w miarę skuteczny model.

### 5.  Jaka jest interpretacja parametrów θ? Ile jest takich parametrów dla D cech i K klas?


### 6.  Jaka jest interpretacja parametrów π? Ile jest takich parametrów dla D cech i K klas?
Parametry te to prawdopodobieństwa a priori wystąpienia danej klasy, jest ich K.


### 7.  Jaka jest interpretacja hiperparametru k? Za co odpowiada? Jaka jest jego interpretacja geometryczna? Jak jego wartość wpływa na rozwiązanie?
K to ilość najbliższych sąsiadów, których bierzemy pod uwagę przy klasyfikacji nowej próbki. 
Geometrycznie zwiększamy otoczenie punktu.
Im większe jest k, tym więcej sąsiadów bierzmy pod uwagę. 
Im większe k, tym bardziej skomplikowany jest model.


### 8.  W jaki sposób wyznaczamy sąsiedztwo w modelu k-NN?
Sąsiedztwo stanowią inne punkty, które są położone w jakieś odległości od siebie. Wyznaczamy je licząc odległości między punktami 
(np. za pomocą metryki Minkowskiego). Następnie wybieramy k najbliższych punktów względem danego punktu. 

### 9.  Czy model k-NN jest modelem generującym, czy dyskryminującym? Czy jest to model parametryczny, czy nieparametryczny?
KNN jest modelem dyskryminującym i nieparametrycznym. 

### 10.  Czy model Naive Bayes jest modelem generującym, czy dyskryminującym? Czy jest to model parametryczny, czy nieparametryczny?
Naive Bayes jest modelem generującym i parametrycznym.