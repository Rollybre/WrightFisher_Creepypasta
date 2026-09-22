# Métriques complémentaires : diversité & proximité empirique/modèle

Fiche de référence détaillée pour les pistes évoquées en discussion, en complément
du test d'Ewens-Watterson déjà implémenté (`ewens_watterson.py`) et du fit power-law
/ test KS déjà présents dans `analysis_stats.py`.

Trois familles :
1. **Diversité intra-distribution** — caractériser empirique et modèle séparément.
2. **Proximité empirique ↔ modèle** — quantifier l'écart entre les deux.
3. **Robustesse de l'hypothèse "loi de puissance"** — vérifier que le framing
   power-law tient face à des alternatives statistiques plausibles.

Toutes les valeurs numériques ci-dessous ont été calculées sur les vraies données
(`data/fandom_data.csv`, 176 catégories, n=30 901 occurrences de tags) et sur un run
de validation du modèle (n=176 classes, mêmes paramètres et seed=42 que dans
`analysis_stats.py`), pour ancrer les explications dans des chiffres réels plutôt que
des formules abstraites. Script de calcul disponible sur demande si vous voulez le
fichier séparé (pas encore committé, comme convenu pour Ewens-Watterson).

---

## 1. Indices de diversité (intra-distribution)

Ces indices se calculent séparément sur chaque distribution (empirique, modèle),
puis se comparent. Contrairement au test d'Ewens-Watterson (qui teste une hypothèse
de neutralité), ce sont des **statistiques descriptives** : elles ne donnent pas de
p-valeur, mais un résumé chiffré interprétable directement.

### 1.1 Nombres de Hill (profil de diversité)

**Intérêt.** Les indices de diversité classiques (richesse, Shannon, Simpson) sont
en réalité des cas particuliers d'une seule famille, les nombres de Hill. Les
présenter comme un *profil* (une courbe en fonction d'un paramètre q) plutôt que
comme des indices isolés permet de voir **à quel ordre** empirique et modèle
divergent : peut-être qu'ils ont la même richesse mais une évenness différente, ou
l'inverse. C'est strictement plus informatif qu'un seul chiffre.

**Fonctionnement.** Pour une distribution de proportions `p₁,...,p_K` :

```
ᵠD = ( Σᵢ pᵢ^q )^(1/(1-q))          pour q ≠ 1
¹D  = exp( -Σᵢ pᵢ·ln(pᵢ) )          pour q = 1 (limite, = exponentielle de Shannon)
```

Chaque `ᵠD` s'interprète comme un **"nombre effectif de catégories"** : le nombre de
catégories parfaitement équitables (toutes de même fréquence) qui donnerait la même
valeur d'indice. Cas particuliers :
- **q=0** → `ᵠD = K`, la richesse brute (nombre de catégories présentes), insensible
  aux fréquences.
- **q=1** → exponentielle de l'entropie de Shannon, pondère chaque catégorie par sa
  fréquence sans sur-pondérer les dominantes.
- **q=2** → inverse de l'indice de Simpson (`1/Σpᵢ²`), pondère plus fortement les
  catégories fréquentes.
- **q→∞** → `1/max(pᵢ)`, ne dépend plus que de la catégorie dominante (équivalent
  du Berger-Parker, §1.3).

Plus q augmente, plus l'indice est dominé par les catégories fréquentes ; à q=0
toutes les catégories comptent pareil, même les singletons.

**Lien avec l'existant.** q=2 correspond exactement à `1/F_obs`, où F_obs est
l'homozygotie déjà calculée dans `ewens_watterson.py` (F_obs=0.0377 → `²D`=26.5,
valeur retrouvée ci-dessous). Les nombres de Hill généralisent ce point unique en un
profil balayé sur q — un moyen naturel de présenter cette famille de résultats.

**Valeurs obtenues :**

| q | Empirique (K=176, n=30 901) | Modèle (run unique, n=176) |
|---|---|---|
| 0 (richesse) | 176.0 | 98.0 |
| 1 (type Shannon) | 45.5 | 12.8 |
| 2 (type Simpson) | 26.5 | 9.6 |
| 3 | 20.4 | 8.5 |

**Lecture.** L'écart se creuse à mesure que q augmente : le modèle n'est pas
seulement moins riche (98 catégories représentées sur 176 dans ce run), il est aussi
nettement moins équitable parmi les catégories qu'il conserve — l'effectif "au sens
Shannon" tombe de 45,5 à 12,8, soit un facteur ~3,5. C'est un chiffre plus parlant
que Δα=0.26 pour un lectorat non spécialiste des lois de puissance.

**Précaution.** Le run "modèle" ici est un run unique (comme celui utilisé pour la
Figure `loglog_fit.png`), pas la moyenne sur 20 runs de `loglog_model_avg.png`. La
richesse (q=0) en particulier est sensible à la variance d'échantillonnage d'un run
individuel — à recalculer sur la moyenne des 20 runs pour un chiffre plus robuste.

**Implémentation** (pas de librairie dédiée nécessaire) :
```python
def hill_number(counts, q):
    p = counts / counts.sum()
    p = p[p > 0]
    if q == 1:
        return np.exp(-np.sum(p * np.log(p)))
    return np.sum(p ** q) ** (1 / (1 - q))
```

---

### 1.2 Coefficient de Gini

**Intérêt.** Mesure d'inégalité universellement connue hors du champ des lois de
puissance (économie, écologie) — facilite la lecture par un lectorat non
spécialiste. Contrairement à α (sensible au choix de xmin et à la partie *tail*
seulement), le Gini résume l'inégalité sur **l'ensemble** de la distribution.

**Fonctionnement.** Construction géométrique : on trie les catégories par fréquence
croissante, on trace la courbe de Lorenz (part cumulée de fréquence en fonction de
la part cumulée de catégories), et G est deux fois l'aire entre cette courbe et la
diagonale d'égalité parfaite (où x % des catégories porteraient x % des occurrences).
G=0 → toutes les catégories égales ; G→1 → une seule catégorie concentre tout.

Formule fermée sur données triées `x` croissant (`i` = 1..n) :
```
G = ( 2·Σᵢ(i·xᵢ) / (n·Σxᵢ) ) - (n+1)/n
```

**Valeurs obtenues :** Gini empirique = **0.796**, Gini modèle = **0.902**.

**Lecture.** Le modèle est *plus* inégalitaire que l'empirique dans ce run, ce qui
va dans le même sens que le §1.1 (richesse effondrée à 98/176) mais est en tension
apparente avec le fait que α_mod (1.27) < α_emp (1.53) — un exposant de loi de
puissance plus faible évoquerait normalement une queue plus "lourde", donc a priori
plus égalitaire parmi les grands rangs. La résolution de cette apparente
contradiction : α ne mesure que la forme dans la région où la loi de puissance
s'applique (au-delà de xmin), alors que le Gini intègre aussi l'effondrement de
richesse (catégories tombées à zéro), qui n'entre pas dans le fit power-law. C'est
un exemple concret de pourquoi combiner plusieurs indices donne une image plus
complète qu'un seul exposant.

**Implémentation :** formule ci-dessus, pas de fonction `scipy` native.

---

### 1.3 Indice de Berger-Parker

**Intérêt.** La mesure de dominance la plus simple et la plus robuste : la part de
la catégorie la plus fréquente. Déjà évoqué qualitativement dans le texte
("*Beings*" ≈ 10 %, l.183) ; le formaliser permet une comparaison chiffrée directe
empirique vs modèle, éventuellement avec un intervalle de confiance bootstrap.

**Fonctionnement.** `d = max(nᵢ) / N`. Le nombre effectif associé est `1/d` (c'est
la limite q→∞ des nombres de Hill, §1.1).

**Valeurs obtenues :** dominance empirique = **10.2 %** (`1/d`=9.8 — cohérent avec
"*Beings*" à 3 158/30 901 ≈ 10.2 %, l.183 du texte), dominance modèle = **17.8 %**
(`1/d`=5.6). La catégorie dominante du modèle capte presque deux fois plus de masse
relative que celle de l'empirique.

**Implémentation :** une ligne (`counts.max() / counts.sum()`).

---

### 1.4 Évenness de Pielou

**Intérêt.** Sépare "combien de catégories existent" (richesse) de "comment les
fréquences sont réparties entre elles" (évenness). Utile ici car la richesse du
modèle chute artificiellement (98/176 dans ce run) pour des raisons de dynamique de
dérive plutôt que de répartition — Pielou permet de juger l'équité *à richesse
observée*, indépendamment de cet effondrement.

**Fonctionnement.** `J = H / ln(K)`, avec `H = -Σpᵢ·ln(pᵢ)` (entropie de Shannon) et
K le nombre de catégories non nulles. J ∈ [0,1], 1 = toutes les catégories
présentes à fréquence strictement égale.

**Valeurs obtenues :** J empirique = **0.738**, J modèle = **0.556**. Même en ne
comptant que les 98 catégories effectivement présentes dans le modèle, leur
répartition entre elles est nettement moins équitable que celle des 176 catégories
empiriques — confirme que l'écart n'est pas qu'un effet de richesse.

**Implémentation :** directe en `numpy`, à partir de H déjà calculé pour Hill q=1.

---

## 2. Mesures de proximité empirique ↔ modèle

Alternatives ou compléments au test KS actuellement utilisé dans `analysis_stats.py`
(`ks_2samp` sur `emp_prop_176` vs `valid_prop_176`, D=0.68, p≈0).

### 2.1 Divergence / distance de Jensen-Shannon (JSD)

**Intérêt.** Le KS ne regarde que l'écart maximal entre les deux fonctions de
répartition cumulées triées — un seul point de la distribution détermine le
résultat. La JSD résume au contraire l'écart **global**, catégorie par catégorie,
entre deux distributions de probabilité. Elle est symétrique (contrairement à une
divergence de Kullback-Leibler simple) et reste finie même quand une catégorie est
à 0 dans une des deux distributions (fréquent ici : 78 catégories à 0 dans le
modèle) — un point où le KL diverge vers l'infini.

**Fonctionnement.**
```
JSD(P,Q) = ½·KL(P‖M) + ½·KL(Q‖M),   avec M=(P+Q)/2 et KL(P‖Q)=Σpᵢ·ln(pᵢ/qᵢ)
```
`scipy.spatial.distance.jensenshannon` retourne en fait `√JSD`, une vraie **distance**
métrique (inégalité triangulaire respectée), bornée par `√(ln 2) ≈ 0.832`.

**Valeur obtenue (comparaison appariée par indice de catégorie) :** distance JS =
**0.461** (divergence sous-jacente = 0.212, sur un maximum théorique de ln2=0.693).
Un écart substantiel — plus de la moitié de la distance maximale possible entre deux
distributions sur ce support.

**Implémentation :** `scipy.spatial.distance.jensenshannon(p_emp, p_mod, base=np.e)`.

---

### 2.2 Distance de Wasserstein (Earth Mover's Distance)

**Intérêt.** Mesure "combien de masse déplacer, et sur quelle distance" pour
transformer une distribution en l'autre. Plus informative que le KS pour des
distributions à longue traîne car elle pondère par l'**ampleur** des écarts et pas
seulement par leur position — deux distributions peuvent avoir le même D de
Kolmogorov-Smirnov mais des masses très différemment déplacées.

**Fonctionnement.** Coût de transport optimal (aire entre les CDF, en dimension 1)
entre les deux distributions vues comme des masses réparties le long d'un axe — ici
l'axe des rangs de catégorie (1 à 176).

**Valeur obtenue :** **8.11** rangs. En moyenne pondérée par les masses de
probabilité, il faut déplacer chaque unité de masse d'environ 8 positions de rang
pour passer de la distribution empirique à la distribution modèle — une échelle
directement interprétable (sur 176 rangs possibles).

**Implémentation :** `scipy.stats.wasserstein_distance(ranks, ranks, u_weights=p_emp, v_weights=p_mod)`.

---

### 2.3 Comparaison appariée (corrélation / régression log-log)

**Intérêt.** Dans `analysis_stats.py`, l'indice de classe du modèle de validation
(n=176) correspond **par construction** au rang empirique
(`init_probs=emp_probs_176` — la classe 0 est initialisée avec la probabilité de la
catégorie empirique la plus fréquente, etc.). Chaque catégorie a donc un vis-à-vis
direct dans le modèle : la comparaison n'a pas besoin d'être traitée comme deux
échantillons indépendants (ce que fait le KS actuel). Une comparaison **appariée**
répond à une question plus fine : "le modèle met-il la bonne fréquence sur la bonne
catégorie", pas seulement "la forme globale est-elle comparable".

**Fonctionnement.** Nuage de points (fréquence empirique, fréquence modèle) par
catégorie, en échelle log-log :
- **Spearman ρ** (sur les rangs, robuste aux valeurs extrêmes et aux zéros) —
  répond à "l'ordre relatif des catégories est-il préservé".
- **Pearson r** sur les log-fréquences (catégories non nulles dans les deux
  distributions) — plus sensible à la forme précise de la relation.
- **Régression log-log** (`log(f_modèle) ~ a·log(f_empirique) + b`) : une pente
  proche de 1 avec R² élevé indiquerait un bon accord proportionnel catégorie par
  catégorie, pas seulement "en moyenne" sur la distribution triée.

**Valeurs obtenues :**
- Spearman ρ = **0.785** (p≈4×10⁻³⁸, n=176) — l'ordre des catégories est
  globalement bien préservé.
- Pearson r (log-log, n=98 catégories non nulles dans les deux) = **0.784**
  (p≈1×10⁻²¹).
- Régression log-log : pente = **1.47**, intercept = -2.79, **R²=0.615**.

**Lecture.** La corrélation est forte et significative — le modèle "sait" quelles
catégories devraient être grandes ou petites, ce qui valide qualitativement le choix
d'initialiser par les proportions empiriques. Mais la pente de 1.47 (≠1) montre que
le modèle **amplifie** les écarts relatifs au-delà de la proportionnalité : une
catégorie deux fois plus fréquente empiriquement se retrouve plus que deux fois plus
fréquente dans le modèle. C'est cohérent avec le mécanisme de dérive (les grandes
classes sont favorisées de façon multiplicative au fil des générations) et donne une
lecture complémentaire au Δα déjà rapporté : ce n'est pas qu'un décalage de pente
globale, c'est aussi une amplification catégorie par catégorie.

**Implémentation :** `scipy.stats.spearmanr`, `scipy.stats.pearsonr`,
`scipy.stats.linregress`.

---

### 2.4 Coefficient de concordance de Lin (CCC)

**Intérêt.** Une simple corrélation tolère un biais systématique (ex. le modèle
toujours au-dessus de l'empirique, comme la pente 1.47 ci-dessus le suggère) tant
que l'ordre relatif est respecté. Le CCC pénalise à la fois le manque de corrélation
**et** le décalage systématique (biais d'échelle ou d'offset) — pertinent car la
question posée n'est pas "les catégories sont-elles dans le bon ordre" (déjà répondu
par Spearman) mais "le modèle reproduit-il les *fréquences*, pas seulement leur
ordre".

**Fonctionnement.**
```
CCC = (2·ρ·σₓ·σᵧ) / (σₓ² + σᵧ² + (μₓ-μᵧ)²)
```
où ρ est la corrélation de Pearson, σ les écarts-types, μ les moyennes. Le
dénominateur ajoute la différence des moyennes au carré, ce qui fait chuter le CCC
en cas de biais systématique même si ρ reste élevé. CCC=1 → accord parfait (tous les
points sur la diagonale y=x, pas seulement alignés).

**Valeur obtenue (sur log-fréquences, catégories non nulles) :** CCC = **0.629**,
nettement plus bas que le Pearson r=0.784 sur les mêmes données — l'écart entre les
deux confirme qu'une partie non négligeable du désaccord vient d'un biais
systématique (la pente 1.47, l'amplification du modèle) plutôt que d'un manque de
corrélation pure.

**Implémentation :** pas de fonction `scipy` native, formule à coder à la main
(quelques lignes, à partir des moyennes/variances/covariance des deux vecteurs).

---

## 3. Robustesse de l'hypothèse "loi de puissance"

### 3.1 Comparaison de vraisemblance power-law vs alternatives

**Intérêt.** Avec seulement 176 points, une loi de puissance est statistiquement
difficile à distinguer à l'œil d'une log-normale ou d'une power-law tronquée
(Clauset, Shalizi & Newman 2009 — référence classique sur ce point, à citer si vous
ajoutez cette analyse). Un·e relecteur·rice familier·ère des lois de puissance
pourrait légitimement demander cette vérification avant d'accepter le framing
"loi de puissance" central au résumé et aux résultats.

**Fonctionnement.** Pour une distribution donnée, on ajuste par MLE plusieurs
alternatives (log-normale, exponentielle, power-law tronquée, stretched
exponential), puis on calcule un rapport de vraisemblance normalisé R = log(L_pl/L_alt)
et sa significativité p (test de Vuong). R>0 avec p<0.05 favorise significativement
la loi de puissance ; R<0 avec p<0.05 favorise significativement l'alternative ; p
non significatif = les données ne permettent pas de trancher entre les deux modèles.

**Valeurs obtenues (empirique, 176 catégories, même fit que celui déjà utilisé pour
α_emp) :**

| Comparaison | R | p | Conclusion |
|---|---|---|---|
| power-law vs exponentielle | +83.65 | <0.0001 | power-law nettement favorisée |
| power-law vs log-normale | −3.97 | 0.0535 | **non tranché** (borderline, p juste au-dessus de 0.05) |
| power-law vs power-law tronquée | −6.98 | 0.0002 | **power-law tronquée significativement favorisée** |
| power-law vs stretched exponential | −4.70 | 0.0448 | stretched exponential légèrement favorisée |

**Lecture — point d'attention pour le papier.** Deux résultats méritent réflexion
avant décision éditoriale :
1. Le test contre la **log-normale** est à la limite de la significativité
   (p=0.0535) : on ne peut pas affirmer avec confiance que la loi de puissance est
   un meilleur modèle qu'une log-normale pour ces données. C'est une réserve
   standard dans la littérature sur les lois de puissance empiriques, à mentionner
   ou nuancer plutôt qu'à ignorer si vous formalisez cette analyse.
2. La **power-law tronquée** est significativement préférée à la power-law pure
   (p=0.0002) — ce qui est assez attendu pour un corpus fini (176 catégories, effet
   de bord en fin de distribution) et n'invalide pas le message principal du papier,
   mais suggère qu'un modèle à troncature serait une description légèrement plus
   fidèle que la power-law simple actuellement rapportée.

Ces résultats ne remettent pas en cause l'essentiel de vos conclusions (le contraste
power-law vs exponentielle reste très net), mais donnent des munitions pour anticiper
une objection de relecture, ou pour nuancer la formulation ("distribution à queue
lourde, compatible avec une loi de puissance" plutôt qu'une affirmation plus forte).

**Implémentation :** déjà disponible via la librairie `powerlaw` utilisée dans
`analysis_stats.py` — `fit.distribution_compare('power_law', 'lognormal')`, etc.
Ajout minimal au script existant (pas de nouveau fichier nécessaire), en réutilisant
l'objet `fit_emp` déjà calculé.

---

## Récapitulatif

| # | Mesure | Répond à | Valeur obtenue (empirique / modèle) | Effort d'implémentation |
|---|--------|----------|--------------------------------------|--------------------------|
| 1.1 | Nombres de Hill | Diversité à différents ordres | q=1: 45.5 / 12.8 | Très faible |
| 1.2 | Gini | Inégalité, lisible hors du champ power-law | 0.796 / 0.902 | Faible |
| 1.3 | Berger-Parker | Dominance de la catégorie n°1 | 10.2 % / 17.8 % | Trivial |
| 1.4 | Pielou (évenness) | Répartition indépendamment de la richesse | 0.738 / 0.556 | Très faible |
| 2.1 | Jensen-Shannon | Écart global empirique/modèle | distance = 0.461 | Trivial (scipy) |
| 2.2 | Wasserstein | Ampleur du déplacement de masse | 8.11 rangs | Trivial (scipy) |
| 2.3 | Corrélation/régression appariée | Bonne catégorie, bonne fréquence ? | ρ=0.785, R²=0.615, pente=1.47 | Faible |
| 2.4 | CCC de Lin | Corrélation + biais systématique | 0.629 | Faible |
| 3.1 | `distribution_compare` | Robustesse du claim "loi de puissance" | lognormal: p=0.054 (limite) | Très faible (déjà dans `powerlaw`) |

**Recommandation de priorité** si vous ne deviez en retenir que deux ou trois pour
le papier : (2.3) la comparaison appariée, qui exploite une propriété déjà présente
dans votre pipeline et raconte une histoire plus fine que le KS seul ; (1.1)/(1.2)
les indices de diversité (Hill/Gini), pour donner des chiffres lisibles par un
public non spécialiste en complément de α ; et (3.1), pour anticiper une objection
de relecture sur le statut "loi de puissance" plutôt que de la découvrir après coup.
