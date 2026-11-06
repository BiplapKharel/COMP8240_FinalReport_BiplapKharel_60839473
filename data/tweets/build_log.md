# Christmas tweets: build log

| Step | Count |
|---|---|
| tweets read (2014) | 50000 |
| dropped: no sentiment emoji | 57160 |
| dropped: advert-like | 4351 |
| dropped: retweet | 27513 |
| dropped: duplicate | 18897 |
| dropped: does not mention Christmas/Xmas | 7533 |
| dropped: both positive and negative emoji | 203 |
| dropped: under 4 words after cleaning | 136 |
| tweets read (2017) | 72203 |
| labelled pool: positive | 4802 |
| labelled pool: negative | 1608 |

Sample: 5000 tweets = all 1608 negative + 3392 randomly chosen positive (seed 2026).

| Split | Tweets | Negative | Positive | 2014 | 2017 |
|---|---|---|---|---|---|
| test | 1000 | 322 | 678 | 531 | 469 |
| dev | 500 | 161 | 339 | 277 | 223 |
| train | 3500 | 1125 | 2375 | 1820 | 1680 |
