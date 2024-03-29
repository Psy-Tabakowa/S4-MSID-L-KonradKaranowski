
# Lista 3 – Regresja logistyczna


### 1.  Wyznaczyć pochodną sigmoidalnej funkcji logistycznej. Zapisać ją jedynie przy pomocy wartości funkcji sigmoidalnej σ(a).
Funkcja sigmoidalna ma postać:
<br>
![](https://cdn.mathpix.com/snip/images/RN7iHSNmfdNp1Krr-aVO-9F5upg0XWrigVR1VOWZ9wY.original.fullsize.png)

Pochodną można policzyć w banalny sposób korzystając z chain rule:
<br>
![](https://cdn.mathpix.com/snip/images/DzJMWMcl9Q_znXGqsUY5bmtrTBpWD-x3px1QDR5Aap4.original.fullsize.png)


Otrzymujemy ostateczną postać:
<br>
![](https://cdn.mathpix.com/snip/images/xVGGL0mQGq078qAgTkPw3tRxnhfUyAmM2vlHqnuNDkQ.original.fullsize.png)

### 2.  Wyznaczyć gradient funkcji celu lub gradient funkcji celu z regularyzacją.
Na początku bez regularyzacji:
Wprowadźmy sobie oznaczenie, które pomoże nam uczynić rachunki bardziej czytelnymi:
<br>
![](https://cdn.mathpix.com/snip/images/rPuiU7Bf2Coispmo7pe_RlQBXN-NNwEi-82A6RSqzlU.original.fullsize.png)

Przyda się też policzona już pochodna sigmoidu:
<br>
![](https://cdn.mathpix.com/snip/images/Tpy-gG0RdFc9To2im0gjzYVvmoZcPduMjxXQ3vd2WnQ.original.fullsize.png)

Możemy policzyć pochodną cząstkową sigmoidu po j-tej wadze, korzystając z naszego poprzedniego rozwiązania:
<br>
![](https://cdn.mathpix.com/snip/images/y0OOVz7NaUySL9UUlRjLw3p2YxQjrzwIvt4F361AGOc.original.fullsize.png)

Nasza funkcja kosztu jest dana wzorem:
<br>
![](https://cdn.mathpix.com/snip/images/k3UBm7BNxePWKSsDXKrQyCkgjEJCM1o-CY4iXFSD-vE.original.fullsize.png)

Liczymy podchodną cząstkową po j-tej wadze:
<br>
![](https://cdn.mathpix.com/snip/images/JSF2aS_xReVZQePBQLs68g29pD8zar1H0J8nGW0AISE.original.fullsize.png)

Rozbijmy to na dwie części:
<br>
![](https://cdn.mathpix.com/snip/images/KmHtjxYzzwKjiwkYFUEAuXyHj3C1qrAf8s6L6M90RiA.original.fullsize.png)
<br>
![](https://cdn.mathpix.com/snip/images/K_oxjHh3rp1KZgqnOvNhfmwG4J3U4xiX0jMcMhIYiCQ.original.fullsize.png)

Całość podstawiamy do wzoru i przekształcamy:
<br>
![](https://cdn.mathpix.com/snip/images/VLaP_VzpSKIeXRrsKsa5IQ4e_3D36AeBPWOVxczR9cg.original.fullsize.png)

Gradient to już se każdy zapisze i przekształci.

Dla regresji z regularyzacją funkcja kosztu wygląda następująco:
<br>
![](https://cdn.mathpix.com/snip/images/tOWGZf-I9FI_tB769YAIBN1oS2t2vd9698jr21DA16g.original.fullsize.png)

Liczymy pochodną cząstkową po j-tej wadze:
<br>
![](https://cdn.mathpix.com/snip/images/E15QFV7NidpHDmVXF9nQMjLvTlLPcFIiNNwGfa9Vovw.original.fullsize.png)

Wprowadźmy oznaczenie:
<br>
![image](https://cdn.mathpix.com/snip/images/K1qCjAlytlg6ZeuMq-jKNeMoy15CcoD1EatCe6fv6BI.original.fullsize.png)

Wynik:
<br>
![](https://cdn.mathpix.com/snip/images/qCHBYtHfZG-zSpElZ5frrT0ETebrpvtWbEOEl9WTy5I.original.fullsize.png)

To się da łatwo przekształcić do gradientu.

### 3.  Co to jest model regresji logistycznej? W jaki sposób modeluje warunkowe prawdopodobieństwo?
Aby zrozumieć regresję logistyczną, należy wprowadzić pojęcie ilorazu szans (odds ratio). Wyrażamy je jako stosunek prawdopodobieństwa pozytywnego zdarzenia (np. pacjent ma raka) do prawdopodobieństwa zdarzenia negatywnego:
<br>
![](https://cdn.mathpix.com/snip/images/p6Ys96ZlsjacHloA5JHCNYqBNF_g9MCR6vWraqzj094.original.fullsize.png)

W praktyce stosuje się tzw. funkcję logitową: (0, 1) => R:
<br>
![](https://cdn.mathpix.com/snip/images/DHqOMb8vtY-F4SLNHw18h5R2QDEVhQP9btEIQFUn4LE.original.fullsize.png)


Funkcja ta jest odwracalna. Jej odwrotnością jest funkcja sigmoid σ: R => (0, 1):
<br>
![image](https://cdn.mathpix.com/snip/images/RN7iHSNmfdNp1Krr-aVO-9F5upg0XWrigVR1VOWZ9wY.original.fullsize.png)


Funkcja sigmoidalna pozwala nam policzyć prawdopodobieństwo warunkowe wystąpienia zdarzenia pozytywnego pod warunkiem wektora cech x, z parametrem w:
<br>
![](https://cdn.mathpix.com/snip/images/c4ZhBqGlglb3ZHoRoPI0_feQgHYJSmKW0LG1oULKHV4.original.fullsize.png)


Regresja logistyczna jest więc metodą, która pozwala nam modelować prawdopodobieństwo przynależności próbki x do klasy pozytywnej, pod warunkiem próbki x.

### 4.  Za co odpowiada wartość progowa θ? W jaki sposób systematycznie podejść do ustalenia jej wartości?
Wartość progowa mówi nam od jakiego prawdopodobieństwa klasyfikujemy próbkę jako przynależną do klasy 1. Jej wartość należy ustalać tzw. metodą Napałowa*

\* patrz. Andriej Napałow - słynny XIX wieczny rosyjski matematyk, znany głównie z odkrycia rewolucyjnej metody rozwiąznywania problemów nazwanej od jego imienia metodą Napałowa. Polega ona na sprawdzeniu wszystkich rozwiązań "na pałę" i wybraniu satysfakcjonującego.


### 5.  Co to jest miara F-measure? Do czego jest wykorzystywana w powyższym zadaniu? Dlaczego zamiast niej nie stosuje się tutaj zwykłej poprawności klasyfikacji na ciągu walidacyjnym?
F measure to metryka służąca do oceny modelu, jest ona średnią harmoniczną dwóch wielkości: 
* precision - mówi o tym, jak precyzyjnie nasz model wykrywa próbki pozytywne: 
<br>
  
![](https://cdn.mathpix.com/snip/images/LRJ0HBwLsWtSsMUKqms07YH12t_rDieDalwVU6GoJeI.original.fullsize.png)
  

* recall - mówi o tym, jak dużo próbek pozytywnych nie zostało wykrytych przez nasz model:
<br>

![](https://cdn.mathpix.com/snip/images/XX5PrmEB2rCDoAEStLxj3ODr7Dw8wEoR6QMqxeUmc4E.original.fullsize.png)
  

Gdzie:
TP - ilość próbek sklasyfikowanych jako pozytywne (w rzeczywistości będących pozytywnymi)
FP - ilość próbek sklasyfikowanych jako pozytywne (w rzeczywistości będących negatywnymi)
FN - ilość próbek sklasyfikowanych jako negatywne (w rzeczywistości będących pozytywnymi)

Same F measure jest dane wzorem:
<br>
![](https://cdn.mathpix.com/snip/images/HpItxXx8ae9A4-h-YLY-ljn18XeLConKMpMjHqHUTLc.original.fullsize.png)

Metrykę tą stosujemy, ponieważ jest to dokładniejsza metoda oceny modelu, przykładowo:

Mamy obraz 1 x 512 x 512 pikseli, na którym chcemy wykryć pieska poprzez zaklasyfikowanie każdego piksela do klasy 1 (ten piksel należy do pieska) albo 0 (ten piksel nie należy do pieska). Sam piesek jednak składa się tylko z ok. 13 000 pikseli. Możemy więc stworzyć model, które bezmyślnie każdy piksel oznacza jako 0. Osiągamy wtedy accuracy na poziomie 0.95. Uradowani biegniemy i chwalimy się naszym osiągnięciem ze światem, po czym okazuje się, że nasz model w rzeczywistości jest okropny. Przykładowo dla tego modelu miara F1 będzie równa... 0.0.



### 6.  Za co odpowiada η w algorytmie gradientu prostego i stochastycznego gradientu prostego? Jak algorytmy będą zachowywać się dla różnych wielkości tego parametru?
η to tak zwany "wspołczynnik uczenia" (learning rate). Mówi nam o tym, jak mocno aktualizujemy wagi.
<br>
![image](https://mvanderbroek.com/images/fastai-lesson2/learning_rate.png)

Dla zbyt dużych wartości η możemy przeskakiwać nad minimum globalnym (przykład 3), dla zbyt małych wartości η nasz algorytm będzie się uczył bardzo długo (przykład 1). Dla dobrze dobranego współczynnika η, nasz model będzie zbiegać w sensownym czasie (przykład 2).


### 7.  Na czym polega detekcja obiektu na zdjęciu? Dlaczego jest to problem klasyfikacji?
Detekcja obiektu na zdjęciu polega w naszym przypadku na przejrzeniu wybranych fragmentów obrazu i zakwalifikowaniu go jako zawierającego obiekt - 1 albo nie zawierającego obiektu - 0.


### 8.  Dlaczego algorytm stochastycznego gradientu prostego zbiega znacznie szybciej? Jakie jest znaczenie wielkości mini-batcha dla zbieżności algorytmu?  Jak będzie zachowywał się dla małych mini-batchy, a jak dla dużych?
W algorytmie stochastycznego spadku wzdłuż gradientu zbiega szybciej, ponieważ wagi są częściej aktualizowane oraz unikamy minimów lokalnych. Rozmiar mini-batcha jest ważny dla szybkości osiągnięcia zbieżoności przez algorytm oraz na ilość epok potrzebnych do osiągnięcia zbieżności. Przykładowo dla dużego batcha, będziemy aktualizować wagi rzadziej, więc algorytm będzie zbiegał wolniej, natomiast dla bardzo małych batchy, algorytm będzie zbiegać szybciej, ale wykonamy znacznie więcej obliczeń. 


### 9.  W jaki sposób można dodać regularyzację L2 na parametry modelu regresji logistycznej? Jaki efekt wówczas osiągniemy? Kiedy konieczne jest stosowanie regularyzacji, a kiedy nie?
Regularyzację L2 dodajemy w następujący sposób:
<br>
![](https://cdn.mathpix.com/snip/images/tOWGZf-I9FI_tB769YAIBN1oS2t2vd9698jr21DA16g.original.fullsize.png)

Regularyzacja polega na penalizowaniu zbyt dużych wag, przez co redukujemy overfitting. Regularyzację trzeba stosować, gdy nasz model się właśnie overfittuje. Aby dodać regularyzację należy najpierw wystandaryzować dane.

### 10.  Na czym polega procedura selekcji modelu w tym zadaniu? Jakie hiperparametry wyznaczamy? Które z nich wymagają każdorazowego nauczenia modelu, a które nie i dlaczego?
W naszym przypadku szukamy najlepszych wag oraz najlepszej wartości progowej θ. W celu wyznaczenia najlepszych wag trenujemy modele na różnych regularyzacyjnych lambdach, a następnie dla wszystkich wytrenowanych wag sprawdzamy, dla jakiej wartości progowej θ osiągamy najlepsze rezultaty. Dla n różnych lambd i m różnych thet musimy wytrenować model tylko n razy (wartość progowa to tylko nasz sposób interpretacji wyników modelu, model ma ją gdzieś). 

