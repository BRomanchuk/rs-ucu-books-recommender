## HW-1 

### Baseline Recommender
[source code](src/recommenders/baseline_recommender.py) | [experiment notebook](experiments/baseline_experiment.ipynb)

Predicts mean book rating regardless of the user.


### PageRank 
[source code and experiment](experiments/page_rank_recommender.ipynb)

Experiment with PageRank implementation for book-crossing dataset. Uses bipartite graph $G(V_I \sqcup V_U, E)$ with users $V_U$ and books $V_I$ as vertices, and rating existence as edges.


## HW-2

### Multi-Armed Bandit Notebooks
[experiment 1](experiments/bandit.ipynb) |
[experiment 2](experiments/mab_recommender.ipynb)

Experiments notebooks that compares epsilon-greedy algorithm, UCB, and Thomas sampling.


## HW-3

### 2-Stage Recommender
[source code and experiment](experiments/two_stage_recommender.ipynb)

2-stage recommender that uses FunkSVD for candidate generation and BPR for ranking.
