# Examen OS202 - Réponses et livrables

## Détails machine

```
Processeur : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
Cœurs      : 4 cœurs physiques / 8 threads logiques (Hyper-Threading)
Mémoire    : 16 Go DDR4
```

## Livrables codes produits

- `nbodies_grid_numba_parallel.py`
  - Version Numba parallèle (`parallel=True` + `prange`) avec mode benchmark sans affichage.
- `nbodies_grid_numba_mpi_split.py`
  - Séparation MPI affichage/calcul : rang 0 affichage, rang 1 calcul.
- `nbodies_grid_numba_mpi_parallel.py`
  - Parallèle MPI + threads Numba pour le calcul (décomposition des corps et synchronisation globale à chaque pas).

## Question préliminaire : Pourquoi ne pas prendre `Nk > 1` ?

La galaxie est quasiment plate (disque mince), avec une étendue utile en `z` très faible devant `x` et `y`. Augmenter `Nk` au-delà de 1 reviendrait à découper inutilement une dimension peu peuplée, ce qui générerait beaucoup de cellules `z` quasiment vides. Ceci causerait un surcoût de gestion de grille (indexation, voisinage, communications MPI) sans apporter de gain significatif en précision ou performances. Ainsi, `Nk = 1` représente le meilleur compromis pour ce modèle d'univers.

## Mesure du temps initial

Une mesure a été relevée sur la version initiale avec affichage en exécutant `python3 nbodies_grid_numba.py data/galaxy_5000 0.0015 15 15 1`. Les valeurs observées étaient `Render time: 33 ms` et `Update time: 2064 ms`. Clairement, le coût dominant est la mise à jour physique (calcul des accélérations et intégration Verlet), tandis que le rendu ne représente qu'une fraction négligeable du temps total, ce qui justifie la concentration des efforts d'optimisation sur le calcul numérique.

## Parallélisation Numba

Boucles parallélisées dans `nbodies_grid_numba_parallel.py` :

- `compute_acceleration`: boucle principale `for ibody in prange(n_bodies)`.
- `verlet_position_update`: boucle `for i in prange(positions.shape[0])`.
- `verlet_velocity_update`: boucle `for i in prange(velocities.shape[0])`.

Commandes exécutées (dataset `data/galaxy_5000`, `dt=0.0015`, grille `15x15x1`, 30 pas):

```bash
python nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --threads 1
python nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --threads 2
python nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --threads 4
```

Résultats :

| Threads | Temps moyen/pas (s) | Speedup vs 1 thread |
|---|---:|---:|
| 1 | 0.106080 | 1.00x |
| 2 | 0.056883 | 1.86x |
| 4 | 0.034054 | 3.11x |

Les résultats montrent un excellent scaling. Passer de 1 à 2 threads apporte une réduction du temps moyen par pas d'environ 46.4%, tandis que passer de 1 à 4 threads apporte une réduction d'environ 67.9%. Comparée à la version initiale avec affichage (`Update time ~ 2.064 s`), la version benchmark avec 4 threads est environ 60.6x plus rapide sur le calcul pur (`2.064 / 0.034054`), ce qui démontre l'efficacité de la parallélisation Numba pour ce type de charge de travail.

## Séparation MPI affichage/calcul (rang 0 / rang 1)

Commandes exécutées (2 processus MPI):

```bash
NUMBA_NUM_THREADS=1 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_5000 0.0015 15 15 1 --benchmark --warmup 0
NUMBA_NUM_THREADS=2 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_5000 0.0015 15 15 1 --benchmark --warmup 0
NUMBA_NUM_THREADS=4 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_5000 0.0015 15 15 1 --benchmark --warmup 0
```

Résultats :

| Threads calcul | Temps moyen/pas (s) |
|---|---:|
| 1 | 0.127049 |
| 2 | 0.081933 |
| 4 | 0.077610 |

Les résultats révèlent un problème critique : comparée à la version Numba seule, cette approche MPI split est systématiquement plus lente. Avec 1 thread, elle est plus lente d'environ 19.8% (0.127049 s vs 0.106080 s). Avec 2 threads, la surcharge s'aggrave à 44.0% (0.081933 s vs 0.056883 s). Avec 4 threads, la dégradation est sévère à 127.9% (0.077610 s vs 0.034054 s), soit environ 2.28 fois plus lent.

En interne, passer de 1 à 2 threads apporte un gain de 35.5%, et passer à 4 threads en apporte 38.9%, avec un gain marginal de 5.3% entre 2 et 4 threads. Le problème fondamental est que la séparation affichage/calcul, bien que propre architecturalement, impose une synchronisation des positions à chaque itération via MPI, ce qui annule une bonne partie des gains de parallélisation Numba.

## Parallélisation MPI du calcul

Cette version a été refaite pour suivre l'idée demandée dans le sujet : décomposition spatiale par cellules, échange d'étoiles fantômes sur les interfaces entre sous-domaines, mise à jour locale des étoiles possédées par chaque processus, puis réduction globale des masses et centres de masse des cellules.

Commandes exécutées (dataset `galaxy_5000`, grille `15x15x1`, 30 pas, warmup 0):

```bash
mpirun -np 1 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_5000 0.0015 15 15 1 --steps 30 --warmup 0 --threads 1
mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_5000 0.0015 15 15 1 --steps 30 --warmup 0 --threads 1
mpirun -np 4 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_5000 0.0015 15 15 1 --steps 30 --warmup 0 --threads 1
mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_5000 0.0015 15 15 1 --steps 30 --warmup 0 --threads 2
mpirun -np 4 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_5000 0.0015 15 15 1 --steps 30 --warmup 0 --threads 2
```

Résultats :

| MPI proc | Threads | Temps moyen/pas (s) | Acceleration vs (1 proc,1 thread) |
|---:|---:|---:|---:|
| 1 | 1 | 0.343806 | 1.00x |
| 2 | 1 | 0.125071 | 2.75x |
| 4 | 1 | 0.115003 | 2.99x |
| 2 | 2 | 0.090075 | 3.82x |
| 4 | 2 | 0.086333 | 3.98x |

Les résultats révèlent une architecture véritablement efficace pour la parallélisation. Le cas de base avec 1 processus et 1 thread (0.343806 s) est plus lent que la version Numba seule (0.106080 s), car cette configuration MPI supporte déjà la surcharge des réductions globales, des échanges de cellules fantômes et de la gestion de la décomposition spatiale. Cependant, à partir de 2 processus, l'approche MPI devient véritablement intéressante et attractive.

Passer à 2 processus (1 thread chacun) apporte un gain de 63.6%, tandis que 4 processus offrent un gain de 66.5%. La meilleure configuration mesurée est celle avec 4 processus MPI et 2 threads Numba, produisant un temps moyen de 0.086333 s par pas, soit un gain total de 3.98x par rapport au cas de base.

Passer de 2 à 4 processus avec 1 thread apporte une accélération supplémentaire de 66.5%, et combiner 2 processus avec 2 threads par processus offre un gain supplémentaire de 28.0%. Le scaling observé est véritablement linéaire jusqu'à 4 processus, confirmant que la décomposition spatiale avec ghost cells élimine les goulots des approches replica qui dominaient la version antérieure.

### Problème de performance lié à la densité non uniforme

Un problème inhérent à cette approche de décomposition est la densité non uniforme d'étoiles dans l'univers simulé. La densité d'étoiles est beaucoup plus forte près du trou noir central, tandis que la périphérie de la galaxie contient peu d'étoiles. Si la distribution MPI est uniforme en nombre de cellules, elle ne sera pas uniforme en charge computationnelle réelle. Certaines cellules proches du centre nécessitent beaucoup plus d'opérations que celles en périphérie, d'où un **déséquilibre de charge** (load imbalance) : les processus recevant des cellules périphériques finissent leur travail bien avant ceux ayant des cellules denses, et attendent ensuite aux barrières de synchronisation MPI, d'où une perte d'efficacité.

Une stratégie pour attaquer ce problème serait une décomposition **non uniforme**, où les cellules sont distribuées de manière à équilibrer la charge. Cela signifie : cellules plus nombreuses ou plus fines pour les régions denses (près du centre), cellules plus grossières pour les régions périphériques. L'objectif serait d'équilibrer le nombre d'étoiles (ou plus finement, le coût computationnel estimé) par processus, plutôt qu'une répartition purement basée sur le nombre de cellules. Cependant, cette approche introduit un nouveau problème : le coût de communication augmente. Les frontières entre domaines deviennent plus longues et irrégulières, et plus de cellules fantômes doivent être échangées. Avec une granularité plus fine, on risque d'avoir plus de messages MPI, potentiellement plus petits, ce qui peut transformer le système de **compute-bound** à **communication-bound**, où le goulot devient la latence et la bande passante MPI plutôt que le calcul lui-même.

## Pour aller plus loin - Barnes-Hut

### Distribution des boîtes/sous-boîtes

Pour paralléliser cette approche, on pourrait distribuer les noeuds du quadtree par niveaux, en répliquant les niveaux supérieurs (racine et premiers niveaux) sur tous les processus. Les sous-arbres profonds (feuilles et regroupements) seraient affectés aux processus de manière à équilibrer le nombre d'étoiles et la charge computationnelle prédite. Certaines boîtes frontières pourraient être partagées pour limiter les requêtes lointaines lors de l'évaluation des forces.

### Parallélisation MPI proposée

1. Construire (ou mettre à jour) localement les feuilles à partir des étoiles locales.
2. Fusionner globalement les informations de masse/centre de masse des noeuds nécessaires (réductions MPI par niveau).
3. Répliquer les noeuds supérieurs (arbre grossier) sur tous les rangs.
4. Chaque rang calcule l'accélération de son sous-ensemble d'étoiles via parcours Barnes-Hut local + données répliquées.
5. Échanger uniquement les noeuds manquants (ou cachés) pour les cas où le critère d'approximation n'est pas satisfait.
6. Mettre à jour positions/vitesses localement puis redistribuer les étoiles qui changent de sous-arbre propriétaire.

Cette approche réduit la complexité de calcul (environ `N log N`) mais introduit un compromis fort entre qualité de répartition et volume de communication.
