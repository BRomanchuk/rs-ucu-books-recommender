# Offline Evaluation Methodology

Performance of the recommender system can be measured in different ways. In the following file we describe the offline evaluation procedure of our algorithm.

## Notations

$U$ - users matrix

$I$ - items (books) matrix

$R$ - ratings matrix

$P$ - predicted ratings matrix

## Metrics

1. Root of Mean Squared Error 

$$L = \sqrt{\frac{1}{|U||I|} \sum_{i \in I} \sum_{u \in U} \left( r_{u, i} - p_{u, i} \right)^2}$$

2. Precision at K

$$P@k = \frac{\text{number of relevant items in the top k positions}}{k}$$

3. Average Precision

$$AP = \sum_{k=1}^m P@k$$

## Train / Test Split

The obvious way to split the data is by the time of the feedbacks. But the records in original ratings table have no timestamps. Moreover they are grouped by users. So, we can try to mimic the time split by random split.

The size of a test split would be 30% of the population.

## Evaluation Procedure

Recommendation algorithm predicts $m$ recommendations for a set of users $U$. It can be either the ordered list of books, or the list of books with ratings.

Then we calculate Average Precision for each user. To evaluate the performance of the recommender, we then calculate the Mean Average Precision among the users.