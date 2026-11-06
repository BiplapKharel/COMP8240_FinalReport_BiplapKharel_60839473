# Human check of the tweet labels

input: data/tweets/human_check_100.csv
tweets: 100

human: no clear feeling 31 (31%); vs emoji label on 69 clear tweets: agreement 82.6%, kappa 0.631
first reading: no clear feeling 55 (55%); vs emoji label on 45 clear tweets: agreement 86.7%, kappa 0.723

human vs first reading:
  3 classes: agreement 62.0%, kappa 0.427
  clear vs none: agreement 62.0%, kappa 0.268
  both gave positive/negative: 38 tweets, agreement 100.0%

confusion (rows = human, cols = first reading):
          0     1  none
      0    14     0    10
      1     0    24    21
   none     3     4    24
