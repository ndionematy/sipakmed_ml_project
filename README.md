<h1 style="color:blue; font-size: 48px;">Cervical Cancer Prediction</h1>

# Description
Le cancer du col de l’utérus est le quatrième cancer le plus fréquent chez les femmes : plus de 530 000 nouveaux cas et plus de 280 000 décès par an. Cette mortalité élevée découle souvent d’un dépistage tardif et d’un accès limité aux tests dans de nombreuses régions.  
Le frottis cervical (test Pap) est un outil de dépistage efficace, mais son analyse manuelle exige une expertise spécialisée et reste sujette aux erreurs humaines.

# Problématique
Comment automatiser de façon efficace et fiable la classification des cellules précancéreuses ou cancéreuses sur les lames de frottis cervicaux tout en réduisant l’intervention manuelle des pathologistes ?

# Objectif
Évaluer l’usage de techniques de classification d’images pour détecter les anomalies cervicales à partir de la base **SIPaKMeD** (4049 images de cellules individuelles réparties en cinq classes), extraite de 966 images de cellules groupées.

# Importance
Intégrer l’intelligence artificielle dans le dépistage précoce du cancer du col de l’utérus :
- soulage les services de santé ;
- sauve des vies grâce au diagnostic précoce — particulièrement dans les zones à ressources limitées.

# Catégories de la base SIPaKMeD
- **Superficial‑Intermediate (SI)** : cellules épithéliales normales superficielles ou intermédiaires.  
- **Metaplastic (M)** : cellules épithéliales glandulaires normales.  
- **Parabasal (P)** : petites cellules souvent présentes chez les femmes ménopausées ; en nombre élevé, peuvent signaler une anomalie.  
- **Koilocytotic (K)** : cellules anormales associées au HPV, considérées comme précancéreuses.  
- **Dyskeratotic (D)** : cellules présentant une kératinisation anormale, souvent liées à des lésions précancéreuses ou cancéreuses.

# Méthodologie

## 1. Préparation des données
- **Chargement** des 4049 images étiquetées
- **Exploration et visualisation** pour examiner la répartition des classes et la qualité des images  
- **Prétraitement** :  
  - redimensionnement (224 × 224 px) ;  
  - normalisation des pixels [0, 1] ;  
  - *data augmentation* : rotation, zoom, retournement, etc.

## 2. Annotation et encodage
- Conversion des étiquettes en valeurs numériques (one‑hot ou label encoding)
- Division du jeu de données :  
  - entraînement 70% ;  
  - validation 10% ;  
  - test 20%.

## 3. Sélection et implémentation du modèle
- Réseau de neurones convolutionnel (CNN) : couches convolutionnelles, pooling, fully connected, ReLU, Softmax.
- Transfer learning* avec DenseNet, VGG16, ResNet50, MobileNet.

## 4. Entraînement
- Fonction de perte appropriée ;
- Optimiseur Adam avec taux d’apprentissage adapté.  
- *Early stopping* et *checkpointing* pour prévenir le surapprentissage.  
- Suivi des fonctions de coût et de l'accuracy

## 5. Évaluation
- Métrique : accuracy, précision, rappel, F1-score,
- Test final : précision globale et par classe, matrice de confusion.  
- Analyse des faux positifs et faux négatifs pour repérer les classes problématiques.  
- Comparaison des performances entre modèles.

## 6. Choix du meilleur modele
- Visualisation des prédictions ;
- Création d'une API pour la prédiction.

# Résultats    
Ces résultats montrent que l’approche par transfer learning avec DenseNet est particulièrement adaptée à ce type de tâche, en permettant d’atteindre des performances élevées avec un temps d’entraînement raisonnable.
Globalement, l'accuracy est de 87.90% sur les données d'apprentissage et 85.93% sur les données de test.
