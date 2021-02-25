# Task 2: Unsupervised Morphological Paradigm Clustering
## Summary
This task is a continuation and subset of the SIGMORPHON 2020 shared task 2. This year, participants will create a computational system that receives raw Bible text, and clusters each token for a given language into morphological paradigms. This is an important first step in building systems that can infer paradigms in an unsupervised fashion.

## Important Links
- [Registration](https://docs.google.com/forms/d/e/1FAIpQLSfdkuA00Uw51KtDNNN_FJgICqOjq2Yx2fPPRzOCU7nb-XZ5YQ/viewform?usp=sf_link)
- [Data](https://github.com/sigmorphon/2021Task2)
- [Baseline](https://github.com/sigmorphon/2021Task2/blob/master/baseline/substring_cluster.py)

## Description
### Unsupervised Paradigm Clustering
In this task, which can be seen as the first step in a pipeline for 2020’s Task 2 on unsupervised morphological paradigm completion, we address paradigm clustering. Given raw text (Bible translations), systems should sort all words into inflectional paradigms.

In future editions, submissions will address other pieces of the pipeline, building upon the paradigm clustering task of this year. For that reason, participants are highly encouraged to submit their system code, which will be made available for participants in future shared tasks that address later parts of the pipeline. The goal is to, over 3 years, develop functional systems for the challenging task of unsupervised paradigm completion.

## Data and Format
For each language, the tokenized Bible (from the JHU Bible corpus) in that language will be provided. Each one will be a separate raw text, utf-8 encoded, file.

### Output Format
The output for each language should contain one line per token. If the same token appears in multiple paradigm clusters due to a process like homophony, it should appear on multiple lines (once per paradigm). Identical tokens within a paradigm due to syncretism, however, do not need to be listed - the evaluation will ignore syncretic forms. Paradigms should be separated by an extra newline, denoting a new paradigm cluster. Also note that our gold standard does not contain any forms consisting of multiple tokens separated by white space. This means that, e.g., the German form `ziehst zusammen`, will not be used in evaluation.

For example, if the tokenized Bible text is:
  `" peace be with you ! I am sending you . "`, then the output format is:

    peace  

    be  
    am  

    with  

    you  

    sent  

    me  
    I  

### Languages
We will release development languages at the start of the shared task. Those languages should be used for model development, hyperparameter tuning, etc. However, performance on the development languages will not be taken into account for the final evaluation. The final evaluation of all submitted systems will be on test languages, which we will only reveal at the beginning of the test phase.

### External Data
In order to enable a fair comparison between systems, we don’t allow the use of any external resources, i.e., anything not provided in the task2 folder of the [data repository](https://github.com/sigmorphon/2021Task2/data). Importantly, this excludes both unlabeled data and any trained models available online. (Thus, the use of pretrained models like morphological analyzers or BERT (Devlin et al., 2018) isn’t allowed!)

## Evaluation
Evaluation will be done on up to 1000 paradigms per language. We will use best-match F1 score, which we compute as follows:
1. Remove all clustered bible tokens that are not in the gold paradigms.
2. Assign each combination of gold and predicted cluster (i.e., paradigm) a score of the number of true positives for the prediction given the gold standard. This is equivalent to the number of overlapping forms between two paradigms.
3. Find the best match between gold and predicted clusters given these scores.
4. Assign every gold and predicted form a class label for the paradigm it belongs to, or is matched with in step 3 (e.g. gold_cluster_1).
  - Assign unmatched paradigms a label representing the spurious cluster it belongs to (e.g. predicted_cluster_1).
5. Compute the F1 score between the resulting labeled gold and predicted forms.

For example, given the above text, if we had an evaluation set of 2 paradigms:
`(be, am) (I, me)` - where both paradigms include only words that occur in the Bible text, we could first evaluate on just the first paradigm `(be, am)`.
Next, we compute the true positives of all found clusters against this paradigm; all result in zero except for the cluster consisting of `be, am`, for which we have 2. So, the best matching pairs up `be, am` and the gold paradigm. We would then evaluate in the same way on the `(I, me)` paradigm, resulting in exactly the same score. We finally label each token according to its matched cluster: `gold_1_be, gold_1_am, gold_2_I, gold_2_me`, and `gold_1_be, gold_1_am, gold_2_I, gold_2_me`, for the predicted and gold words, respectively. A final F1 score is then computed for the set of predicted forms given the set of gold standard forms, resulting in an F1 score of 100%.

The evaluation will be done with the `eval.py` script [here](https://github.com/sigmorphon/2021Task2/blob/master/evaluate/eval.py)

## Baseline
We will compare submissions against a very basic baseline that functions as follows:
Cluster all words together which share a common substring of length `n`, removing any duplicate paradigms that this creates. `n` is a tunable hyperparameter that is chosen using the development languages.

## Bonus Tasks
There are two bonus tasks: *cell clustering*, that is, additionally sorting found inflections into paradigm slots, and *unsupervised morphological paradigm completion*, that is, additionally generating missing forms in a paradigm. Both of those tasks will be evaluated using best-match accuracy (the metric from 2020’s Task 2); we will train a neural string transducer (Makarov and Clematide, 2018) to obtain this for systems which only do the first bonus task (cell clustering). The baseline for the bonus tasks is the 2020 Task 2 baseline.

## Code
We strongly encourage participants to submit their code to us along with their system descriptions. The code will be provided to participants of the next years' shared task, who will be working on the next stage: Paradigm Cell Clustering.

## Timeline
- March 1, 2021: Dev Data released.
- March 1, 2021: Baseline code and results released.
- April 17, 2021: Test Data released.
- May 1, 2021: Participants' submissions due.
- May 8, 2021: Participants' draft system description papers due.
- May 15, 2021: Participants' camera-ready system description papers due.

## Organizers

  Adam Wiemerslage, University of Colorado Boulder  
  Arya McCarthy, Johns Hopkins University  
  Alexander Erdmann, Ohio State University  
  Manex Agirrezabal, University of Copenhagen  
  Garrett Nicolai, University of British Columbia  
  Miikka Silfverberg, University of British Columbia  
  Mans Hulden, University of Colorado Boulder  
  Katharina Kann, University of Colorado Boulder  


## Contact
Please contact us with any questions at adam.wiemerslage@colorado.edu

# This repository

The code in this repository assumes python 3.9 is used. All dependencies are in the requirements.txt file and can be installed with pip, using the command:

```
pip install -r requirements.txt
```
