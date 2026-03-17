# Examen OS202 - Reponses et livrables

## Livrables codes produits

- `nbodies_grid_numba_parallel.py`
  - Version Numba parallele (`parallel=True` + `prange`) avec mode benchmark sans affichage.
- `nbodies_grid_numba_mpi_split.py`
  - Separation MPI affichage/calcul : rang 0 affichage, rang 1 calcul.
- `nbodies_grid_numba_mpi_parallel.py`
  - Parallele MPI + threads Numba pour le calcul (decomposition des corps et synchronisation globale a chaque pas).

## Question preliminaire

**Pourquoi ne pas prendre `Nk > 1` ?**

La galaxie est quasiment plate (disque mince) : l'etendue utile en `z` est tres faible devant `x`/`y`.
Avec `Nk > 1`, on decoupe inutilement une dimension peu peuplee :

- beaucoup de cellules `z` seraient presque vides,
- plus de cout de gestion de grille (indexation, voisins, communications en MPI),
- peu ou pas de gain de precision/performances.

Donc `Nk = 1` est le meilleur compromis dans ce modele.

## Mesure du temps initial

En pratique, sur ce type de code N-corps, la partie la plus interessante a paralleliser est **le calcul des trajectoires/accelerations** (double boucle de forces), car c'est la partie dominante en complexite.

L'affichage est utile mais en general secondaire pour `N` suffisamment grand, et il est souvent limite par le rendu/IO graphique plutot que par l'ALU CPU.

## Paralleisation Numba

Commandes executees (dataset `data/galaxy_1000`, `dt=0.001`, grille `20x20x1`, 12 pas, warmup 2):

```bash
/home/diego/4OS02/.venv/bin/python nbodies_grid_numba_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2 --threads 1
/home/diego/4OS02/.venv/bin/python nbodies_grid_numba_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2 --threads 2
/home/diego/4OS02/.venv/bin/python nbodies_grid_numba_parallel.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2 --threads 4
```

Resultats :

| Threads | Temps moyen/pas (s) | Acceleration vs 1 thread |
|---|---:|---:|
| 1 | 0.011517 | 1.00x |
| 2 | 0.005772 | 2.00x |
| 4 | 0.004953 | 2.33x |

## Separation MPI affichage/calcul (rang 0 / rang 1)

Commandes executees (2 processus MPI):

```bash
NUMBA_NUM_THREADS=1 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2
NUMBA_NUM_THREADS=2 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2
NUMBA_NUM_THREADS=4 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_split.py data/galaxy_1000 0.001 20 20 1 --benchmark --steps 12 --warmup 2
```

Resultats :

| Threads calcul | Temps moyen/pas (s) |
|---|---:|
| 1 | 0.011999 |
| 2 | 0.008509 |
| 4 | 0.007414 |

Comparaison a la version Numba seule :

- 1 thread: legerement plus lent (~4%).
- 2 et 4 threads: nettement plus lent (surcout de communication MPI et synchro a chaque pas).

**Constat** : la separation affichage/calcul est propre architecturalement, mais pour ce cas la communication des positions a chaque iteration annule une bonne partie du gain Numba.

## Paralleisation MPI du calcul

Commandes executees (dataset `galaxy_1000`, grille `20x20x1`, 10 pas, warmup 2):

```bash
NUMBA_NUM_THREADS=1 mpirun -np 1 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --steps 10 --warmup 2
NUMBA_NUM_THREADS=1 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --steps 10 --warmup 2
NUMBA_NUM_THREADS=1 mpirun -np 4 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --steps 10 --warmup 2
NUMBA_NUM_THREADS=2 mpirun -np 2 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --steps 10 --warmup 2
NUMBA_NUM_THREADS=2 mpirun -np 4 /home/diego/4OS02/.venv/bin/python nbodies_grid_numba_mpi_parallel.py data/galaxy_1000 0.001 20 20 1 --steps 10 --warmup 2
```

Resultats :

| MPI proc | Threads | Temps moyen/pas (s) | Acceleration vs (1 proc,1 thread) |
|---:|---:|---:|---:|
| 1 | 1 | 0.011394 | 1.00x |
| 2 | 1 | 0.011468 | 0.99x |
| 4 | 1 | 0.018120 | 0.63x |
| 2 | 2 | 0.010209 | 1.12x |
| 4 | 2 | 0.010178 | 1.12x |

### Probleme de performance lie a la densite non uniforme

La densite d'etoiles est plus forte pres du trou noir, donc certaines zones/cellules font beaucoup plus de calcul que d'autres.

- Si la distribution MPI est uniforme en nombre de cellules, elle n'est pas uniforme en charge.
- On obtient du **desquilibre de charge** (load imbalance): certains processus finissent tard, les autres attendent aux barrieres/synchronisations.

### Distribution "intelligente" proposee

Une bonne strategie est une decomposition non uniforme :

- cellules plus fines et/ou plus nombreuses pour les regions denses,
- cellules plus grossieres et/ou moins nombreuses en peripherie,
- objectif: equilibrer le nombre d'etoiles (ou le cout estime) par processus, pas juste le nombre de cellules.

### Nouveau probleme de performance possible

Avec cette distribution intelligente, le cout de communication peut augmenter :

- frontieres plus longues/irregulieres entre domaines,
- plus de cellules fantomes a echanger,
- granularite plus fine => plus de messages MPI, parfois plus petits,
- donc risque de devenir **communication-bound**.

## Pour aller plus loin - Barnes-Hut

### Distribution des boites/sous-boites

Proposition:

- Distribuer les noeuds du quadtree par niveaux, avec replication des niveaux hauts (racine + premiers niveaux) sur tous les processus.
- Affecter ensuite les sous-arbres profonds (feuilles/regroupements de feuilles) aux processus de facon a equilibrer le nombre d'etoiles et le cout predit.
- Autoriser le partage de certaines boites frontieres pour limiter les requetes a distance pendant l'evaluation des forces.

### Paralleisation MPI proposee (sur papier)

1. Construire (ou mettre a jour) localement les feuilles a partir des etoiles locales.
2. Fusionner globalement les informations de masse/centre de masse des noeuds necessaires (reductions MPI par niveau).
3. Repliquer les noeuds superieurs (arbre grossier) sur tous les rangs.
4. Chaque rang calcule l'acceleration de son sous-ensemble d'etoiles via parcours Barnes-Hut local + donnees repliquees.
5. Echanger uniquement les noeuds manquants (ou caches) pour les cas ou le critere d'approximation n'est pas satisfait.
6. Mettre a jour positions/vitesses localement puis redistribuer les etoiles qui changent de sous-arbre proprietaire.

Cette approche reduit la complexite de calcul (environ `N log N`) mais introduit un compromis fort entre qualite de repartition et volume de communication.
