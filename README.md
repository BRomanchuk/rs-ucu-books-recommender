# rs-ucu-books-recommender
UCU Recommender Systems course project


## Project Structure
```bash
├── artifacts        # artifacts folder (A/B Testing Framework)
├── data             # folder with books, users, and ratings data
├── experiments      # folder of notebooks with all the recommenders testing and Multi-armed bandit simulation
└── src
    ├── entities     # wrapper classes for users, items, and ratings
    ├── recommenders # wrapper classes for recommendation algorithms
    └── evaluators   # evaluation realizations    
```


## Installation

1. Clone repo
```bash
git clone https://github.com/BRomanchuk/rs-ucu-books-recommender.git
```
2. Create and activate virtual environment
```bash
python3.10 -m venv ~/.books-recommender
source ~/.books-recommender/bin/activate
```
3. Install requiremets
```bash
pip install -r requirements.txt
``` 

4. Add python kernel to run experiments notebooks
```bash
python -m ipykernel install --user --name .books-recommender --display-name ".books-recommender"
```

## HW-1

## Exploratory data analysis
[experiment notebook #1](experiments/mykyta_eda.ipynb)
[experiment notebook #2](experiments/bohdan_eda.ipynb)
[experiment notebook #3](experiments/sunnycows_eda.ipynb)

## Content Based Filtering
[source code](src/recommenders/content_recommender.py) | [experiment notebook](experiments/contentbased.ipynb)
Calculates distances between items using tfidf matrix, calculated for textual features

### Baseline Recommender
[source code](src/recommenders/baseline_recommender.py) | [experiment notebook](experiments/baseline_experiment.ipynb)

Predicts mean book rating regardless of the user.

### Item-based Collaborative Filtering Recommender
[source code and experiment notebook](experiments/item_item_lr.ipynb)

Recommendations based on finding similarities between the items in terms of how they are rated by the users

### User-based Collaborative Filtering Recommender
[source code and experiment notebook](experiments/user_user_lr.ipynb)

Recommendations based on finding similarities between the users in terms of how they rate the books

### PageRank
[source code and experiment](experiments/page_rank_recommender.ipynb)

Experiment with PageRank implementation for book-crossing dataset. Uses bipartite graph $G(V_I \sqcup V_U, E)$ with users $V_U$ and books $V_I$ as vertices, and rating existence as edges.


## HW-2

## Alternating least squares
[experiment notebook](experiments/als_recommender.ipynb)
Predictions using ALS recommender with LibRecommender 

## FunkSVD
[experiment notebook](experiments/funksvd_recommender.ipynb)
Predictions using FunkSVD recommender with LibRecommender

## NCF and DNN
[source code](src/recommenders/dnn_recommender.py) | [experiment notebook](experiments/dnn.ipynb)
[experiment notebook](experiments/ncf_recommender.ipynb)

Predictions using custom dnn algo, and with NCF from LibRecommender

### Multi-Armed Bandit Notebooks
[experiment 1](experiments/bandit.ipynb) |
[experiment 2](experiments/mab_recommender.ipynb)

Experiments notebooks that compares epsilon-greedy algorithm, UCB, and Thomas sampling.


## HW-3

## L2R 
[experiment notebook](experiments/l2rWideDeep_recommender.ipynb)  [experiment notebook](experiments/l2rBPR_recommender.ipynb)
Experiments with Wide Deep and BPR from LibRecommender

### 2-Stage Recommender
[source code and experiment](experiments/two_stage_recommender.ipynb)

2-stage recommender that uses FunkSVD for candidate generation and BPR for ranking.

### Sequential Recommender
[experiment notebook](experiments/rnn_lr.ipynb)

RNN/LSTM-based sequential recommender based on embeddings with LibRecommender.

[experiment notebook](experiments/lstm.ipynb)

RNN/LSTM-based sequential recommender based on the order of ratings with augmented timestamps using tensorflow/lstm layer.

## Data
[Kaggle Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset)