
# I built Logistic Regression, Decision Trees, Random Forests, and K-Means from scratch — here's what surprised me

Available on github https://github.com/hariomlohardev/mini-sklearn

KMeans matched sklearn to the decimal while every classifier trailed.

## What I built

`minisklearn`: five from-scratch models plus hand-rolled metrics, all
stdlib Python, trained on the Titanic dataset (889 rows, 7 features,
~38% survived). Same split throughout (`test_ratio=0.2, seed=0`), all
scored with my own confusion-matrix metrics.

## The surprise

All the other models that i tested against the sklearn library are performing a bit bad than sklearn implementation but the KMeans matched exectly each things no differance 
I didn't expect that to happend 

## The numbers

Same split, same ruler — my models vs `scikit-learn`:

| model | mine (acc / P / R / F1) | sklearn (acc / P / R / F1) | gap | verdict |
|---|---|---|---|---|
| LogisticRegression | 0.684 / 0.714 / 0.294 / 0.417 | 0.847 / 0.806 / 0.794 / 0.800 | 0.164 | investigate |
| DecisionTree | 0.706 / 0.682 / 0.441 / 0.536 | 0.780 / 0.716 / 0.706 / 0.711 | 0.073 | investigate |
| RandomForest | 0.695 / 0.652 / 0.441 / 0.526 | 0.831 / 0.828 / 0.706 / 0.762 | 0.136 | investigate |
| KMeans | inertia 742754.6 | inertia 742754.6 | 0.0 | match |

The pattern: my classifiers systematically under-predict "survived"
(recall 0.29–0.44 vs sklearn's 0.71–0.79) 

A too-tiny learning rate on
unscaled features plus exact ID3 splits fragmenting continuous inputs.
KMeans, meanwhile, matches sklearn's inertia to the decimal, which says
the Lloyd loop is exactly right.

## What I'd do differently

In future i will follow this models approch if something can be solved by this rather than jumping to Neural networks and gradient decent 

I know there are problems that can't be solved by this tiny models but they surpries so much well on some special cases where neutral networks perform bad 
