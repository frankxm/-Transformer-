# Prediction de l'orientation des tremblements de terre basée sur Transformer


## Introduction du projet

Parmi les divers événements d'urgence, les tremblements de terre sont les plus connus, se caractérisant par leur portée globale, leur complexité et leur soudaineté. Pour prédire l'orientation des tremblements de terre, des méthodes telles que l'utilisation de piézomètres de haute précision sont couramment employées, permettant d'analyser les variations des données de contrainte avant le séisme. Ce travail propose une approche différente, utilisant le modèle Transformer pour capturer les informations temporelles et spatiales des données multidimensionnelles. Un modèle optimal de prédiction de l'orientation des tremblements de terre est ensuite formé, et une interface utilisateur est conçue pour la visualisation des résultats à l'aide de Tkinter.


## Description des fichiers

### `Earthquake.py`
Ce fichier permet d'analyser et de visualiser des séries temporelles de données de contrainte. Il effectue les opérations suivantes :
- Lecture et fusion de fichiers CSV.
- Détection et correction des valeurs aberrantes.
- Lissage des données à l'aide de moyennes mobiles.
- Analyse des variations pendant les événements sismiques.
Des graphiques détaillés sont générés pour observer les tendances et anomalies dans les données de contrainte.

### `Kmeans.py`
Ce fichier analyse et visualise la répartition des zones sismiques à partir des données de latitude et longitude. Il utilise l'algorithme **KMeans** pour effectuer le clustering et évaluer différentes valeurs de `k` à l'aide des critères suivants :
- SSE (Somme des erreurs au carré)
- Silhouette score
- Indice de Davies-Bouldin

Le fichier génère des visualisations, telles que des courbes SSE, des diagrammes de clusters, et des cartes de dispersion avec les centres des clusters. De plus, il permet de vérifier les étiquettes des points spécifiques et de tracer l'enveloppe convexe des points pour délimiter les zones analysées.

### `Machine_learning.py`
Ce fichier permet de comparer un modèle d'apprentissage profond (Transformer) avec des méthodes d'apprentissage automatique. Les principales étapes incluent :
- Lecture et prétraitement des données.
- Sélection des échantillons et construction des étiquettes.
- Normalisation des données.
- Formation et évaluation du modèle.
- Vérification des performances du modèle, avec des courbes de validation et de test.

### `Earthquake_gui.py`
#### Utilisation :
1. **Étape 1** : Importez les fichiers de données de contrainte de la station, le catalogue des tremblements de terre et le fichier modèle. Si les données de contrainte sont sous format texte brut, elles sont prétraitées pour afficher des informations pertinentes telles que l'heure d'importation, le nom de la station, sa longitude, sa latitude, et le volume des données. De même, après l'importation du catalogue des tremblements de terre, les informations suivantes sont affichées : heure d'importation, zone de données, plages de longitude et de latitude, profondeur et détails du réseau sismique.
2. **Étape 2** : Une fois les trois fichiers importés, vous pouvez cliquer sur le bouton de prédiction pour prédire les tremblements de terre. Les résultats seront affichés sur une carte interactive via Amap, avec la possibilité de visualiser les résultats à votre convenance.

### `Zone_map.py`
Ce fichier utilise la bibliothèque **Folium** pour créer une carte interactive. Il applique l'algorithme **KMeans** pour effectuer une analyse de clustering sur des données géographiques, créant ainsi une carte interactive qui visualise dynamiquement les différentes zones géographiques par clusters. Cette fonctionnalité est essentielle pour l'interface graphique (GUI).

---

## Expérience du modèle Transformer

### `Dataset_process/earthdataset_process.py`
Ce fichier définit la classe `MyDataset` pour créer un ensemble de données personnalisé à l'aide de **PyTorch** pour l'entraînement et le test d'un modèle de machine learning. Il applique des techniques d'augmentation de données sur des séries temporelles.

### `Module/`
Le répertoire définit un modèle **Transformer** comprenant plusieurs composants essentiels :

### module/encoder.py
Le fichier `encoder.py` définit la classe `Encoder`, qui est une partie essentielle du modèle Transformer. L'Encoder utilise plusieurs composants clés pour traiter les données d'entrée et les transformer en une représentation utile pour les étapes suivantes du modèle. Voici les principales fonctionnalités de l'Encoder :

1. **MultiHeadAttention** : L'Encoder applique l'attention multi-tête pour capturer les relations complexes entre les éléments de la séquence d'entrée.
2. **FeedForward** : Après l'attention, les données passent par un réseau de neurones feedforward pour affiner la représentation.
3. **Dropout** : La technique de régularisation `dropout` est utilisée pour éviter le surapprentissage pendant l'entraînement.
4. **Layer Normalization** : Cette technique est utilisée pour normaliser la sortie de chaque couche, stabilisant ainsi l'entraînement.
5. **Residual Connection** : Des connexions résiduelles sont ajoutées pour faciliter le flux des gradients et éviter les problèmes de gradient disparu.




### module/multiheadattention.py
Le fichier `multiheadattention.py` définit la classe `MultiHeadAttention`, qui implémente le mécanisme d'attention multi-tête, un composant clé du modèle Transformer. Ce mécanisme permet au modèle de se concentrer sur différentes parties d'une séquence d'entrée en parallèle, capturant ainsi des relations complexes dans les données. Voici les principales fonctionnalités du module :

1. **Diviser les têtes d'attention** : Chaque tête d'attention capture un aspect différent des relations dans la séquence, comme les relations syntaxiques ou sémantiques.
2. **Calcul de l'attention** : Les requêtes (Q), les clés (K) et les valeurs (V) sont utilisées pour calculer les scores d'attention et obtenir la sortie pondérée.
3. **Masquage** : Un masquage est appliqué pendant l'entraînement pour garantir que le modèle ne voit pas les informations futures dans la séquence.
4. **Parallélisme** : Les têtes d'attention sont calculées en parallèle, ce qui permet au modèle d'apprendre plusieurs relations en même temps.
5. **Softmax** : La fonction Softmax est appliquée aux scores d'attention pour normaliser les poids d'attention.



### module/feedforward.py
Le fichier `feedforward.py` définit la classe `FeedForward`, qui implémente un réseau de neurones entièrement connecté utilisé dans le modèle Transformer. Ce composant permet de traiter les informations après le passage à travers l'attention multi-tête. Il est composé de deux couches linéaires avec une activation ReLU pour capturer des relations non linéaires dans les données. Voici les principales caractéristiques du module :

1. **Couches linéaires** : La première couche transforme l'entrée de dimension `d_model` à `d_hidden`, et la seconde couche ramène la sortie à la dimension `d_model`.
2. **Activation ReLU** : Appliquée après la première transformation pour introduire des non-linéarités dans le modèle, ce qui améliore sa capacité à apprendre des relations complexes.
   
### module/loss.py
Le fichier `loss.py` définit la classe `Myloss`, qui implémente une fonction de perte personnalisée pour le modèle Transformer en utilisant la **Cross-Entropy Loss**. Cette fonction est couramment utilisée pour les tâches de classification multi-classes. Elle est utilisée pour calculer la différence entre les probabilités prédites par le modèle et les étiquettes réelles. Voici les principales caractéristiques de ce module :


### module/transformer.py
Le fichier `transformer.py` définit la classe `Transformer`, qui implémente le modèle Transformer pour prédire des événements d'urgence, comme les tremblements de terre. Le modèle est composé de plusieurs encodeurs (une partie traitant les données par étapes et l'autre par canaux). Voici un résumé des composants et de leur fonctionnement :

1. **Encoders Step-wise et Channel-wise** : Les données sont d'abord traitées par un ensemble d'encodeurs spécialisés pour les données par étapes, puis par un autre ensemble spécialisé pour les données par canaux.
2. **Position Encoding (pe)** : Le modèle peut ajouter un encodage de position pour tenir compte de la séquence des données.
3. **Mécanisme de Gate** : Un mécanisme de gate est utilisé pour pondérer l'importance des encodages traités par les deux ensembles d'encodeurs.
4. **Fusion des encodages** : Les sorties des encodeurs sont fusionnées et passées à travers une couche linéaire pour produire la sortie finale.

Le modèle peut être utilisé pour des tâches de prédiction où il est nécessaire de prendre en compte à la fois les informations temporelles (par étapes) et les informations contextuelles (par canaux). Le mécanisme de gate permet d'adapter dynamiquement l'importance de chaque aspect des données pour la prédiction finale.




### `Run_earthquake.py`
Ce fichier sert à **entraîner et valider le modèle Transformer** en réalisant plusieurs expériences. Il explore la performance du modèle en fonction de divers paramètres, tels que :
- Taux d'apprentissage
- Taille du lot
- Optimiseur
- Nombre d'époques
- Stratégie de décroissance du taux d'apprentissage
- Structure de l'ensemble de données

L'objectif est d'obtenir le modèle optimal avec les meilleures performances.

### `Run_with_saved_model.py`
Ce fichier permet d'évaluer les performances d'un modèle Transformer pré-entraîné sur un jeu de données de test. Il génère des résultats sous forme de métriques et de graphiques visuels pour analyser la performance du modèle dans une tâche de classification multi-classes.

---

## Prérequis
Pour exécuter ce projet, vous devez avoir installé les bibliothèques suivantes :
- **PyTorch**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Seaborn**
- **Matplotlib**
- **Folium**

Vous pouvez installer ces dépendances en exécutant :


pip install torch numpy pandas scikit-learn seaborn matplotlib folium

## Détails supplémentaires
Si vous souhaitez en savoir plus sur les détails techniques et théoriques derrière ce projet, vous pouvez consulter l'article suivant : [Lire l'article ici](https://drive.google.com/file/d/1Ev51Qw1txSkjH7OhbzFTypYUKFZ4OkxN/view?usp=drive_link)


Si vous souhaitez obtenir les données utilisées dans le projet, veuillez cliquer ici:[Obtenir les données ici](https://drive.google.com/drive/folders/1nicOJgUZ_VURFw1weYCK69bnmhH7jzpB?usp=drive_link)


