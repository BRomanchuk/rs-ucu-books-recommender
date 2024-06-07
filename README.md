# rs-ucu-books-recommender
UCU Recommender Systems course project


## Project Structure
```bash
├── artifacts        # artifacts folder
├── data             # folder with books, users, and ratings data
├── experiments      # folder with experiments
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

## Data
[Kaggle Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset)