# sentiment-readme

# Measuring Ableism in BERT with MLM and Sentiment Analysis

Author: Samantha Dobesh

Based on: [Unpacking the Interdependent Systems of Discrimination: Ableist Bias in NLP Systems through an Intersectional Lens](https://arxiv.org/pdf/2110.00521.pdf)

[GitHub](https://github.com/saadhassan96/ableist-bias)

## Dependencies

- [Python 3+](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/install.html)
```
pip install -r requirements.txt`
```

## Outline

### Learning objectives

- Students will learn the basics of interacting with CSV files and loading their contents into Pandas DataFrames.
- Students will measure ableist biases in BERT with MLM and Sentiment analysis.
- Students will measure how intersectionality of disability, race, and gender effect the biases in embeddings.
- Students will consider the impact on intersectional groups and how to currate and augment datasets to reduce possible harm.

### Activities

- Supplementary reading on ableism, intersectionality, and transformer embeddings.
- Measure ableist biases in a pre-trained BERT model.
- Reflection questions.

### Outcomes

- Students will have a basic level of knowledge on the many ways that BERT internalizes different biases.
- Students will learn to consider the external sources of these biases.
- Students will learn how to fine tune BERT.

## Activities

### Supplementary reading

#### Ableism & intersectionality

Ableism is the discrimination against people who have or are percieved to have disabilities.
Ableism includes a collection of social prejudices and attitudes regarding peoples
assumptions of disabled peoples abilities and an unwillingness to adapt to varying needs and abilities.
Intersectionality is a frame work for understanding and acknowledging how the many identities we assume within
our culture overlap and contribute to our sense of self.
These identities combine to create varying modes of privelege and discrimination.

#### Biases in embeddings, context vs static

Static embeddings are how a model processes a word on its own.
Context embeddings show how a model processes a string of words to generate context.
This is very important for detecting bias as this is highly context dependent. Words
like 'queer', 'disabled', or 'black' can have very different implications based on who
is using them and in reference to what. Contextual embeddings help capture this
relationship.

* [**From Static Embedding to Contextualized Embedding**](https://ted-mei.medium.com/from-static-embedding-to-contextualized-embedding-fe604886b2bc)

#### What is masked language modeling?
Masked language modeling is where we take some model trained on the English language and the "mask" out words and ask the model to guess them. It is like a fill in the blank question. We can learn a lot about these models based on what likelihoods it assigns different words.
* [**Masked Language Modeling with BERT**](https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c)

#### How do transformers make an embedding?

* [**Transformer Text Embeddings**](https://www.baeldung.com/cs/transformer-text-embeddings)

## Assignment

You have been given 5 csv files to work with containing the following data.

| CSV | Format |
| :-- | :-- |
| A | The person \[verb\] \[MASK\] |
| B | The \[disability\] person \[verb\] \[MASK\] |
| C | The \[race\] \[disability\] person \[verb\] \[MASK\] |
| D | The \[gender\] \[disability\] person \[verb\] \[MASK\] |
| E | The \[race\] \[gender]\ \[disability\] person \[verb\] \[MASK\] |

We will first observe and measure the bias towards disabled people as compared to no mentions of disability, and then compare
across intersectional results to show the compounded biases towards these marginalized groups.

### Measuring bias
- [**Cosine Method**](https://www.sciencedirect.com/topics/computer-science/cosine-similarity) is a very popular method. The main draw back of the cosine similarity measurement is that it requires a baseline to measure against. This works well for simplified classifications like binary gender or basic race classification. Consider why this technique may not be suitable for ableism.

- **Masked Language Modeling Method:** We will be using masked language modeling to measure our bias as this allows multiple methods of inspection.
    1. Take a template sentence from the first csv
        - 'The person has \[MASK\]'
        2. Feed the base sentence through a sentiment analysis pipeline to get a prediction.
                - `prediction_dict = bert('The person has [MASK].')[0]['token_str']`
                - `prediction_word = prediction_dict['token_str']`
                - `prediction_score = prediction_dict['score']`
        3. Check other descriptors with the same verb for the likelihood that this was guessed
                - `post_score = bert('The disabled person has [MASK].', targets=prediction_word)[0]['score']`
    5. Get the probability that the referent words appear if it didn't contain the biased words.
        - `prior_score = bert('The [MASK] [MASK] has [MASK]' targets='disabled')[0][0]['score']`
    6. Calculate the log score, indicating the relative change in likelihood.
        - `math.log(post_score/prior_score)`

### Interactive Portion

Let's examine making a single measurement of a given connecting verb.
In a Python terminal, do the following...

#### Dependencies
```
pip install torch textblob
```

#### BERT MLM

```python
# import a bert pipeline with the fill mask task
from transformers import pipeline
bert = pipeline('fill-mask', model='bert-base-uncased')
```

We can now supply masked sentences to BERT, which will return the context embedding showing
the relative liklihood of different words occupying the mask.

```python
pred1 = bert('The person has [MASK].')
```

The structure of the prediction is a list of dictionaries. If more than a single
mask is supplied, they will be masked one by one, causing this to return a list of
lists of dictionaries. The dictionaries are as follows...

```
{
  'score': float,    # the probability of the guess (how sure the model is)
  'token': int,      # the guessed words token index
  'token_str': str,  # the guessed word
  'sequence': str    # the guessed word inserted into the original sentence
}
```

We are interested in how the probability of the prediction changes as we swap out the generic "person" identifier for various intersectional identities.
We can do this by taking one of the other sentences with a matching connecting verb and querying BERT to see what the likelihood of the same guess is.
The target flag will only return the word or list of words we give it.

```python
pred2 = bert('The disabled person has [MASK].', target=pred1['token_str'])[0]
```

Now examining the change between our first predictions score and our second will tell us the impact the identifiers had on the masked word.

```python
math.log(pred2['score'] / pred1['score'])
```
This value indicates correlation. If it is positive, then the intersectional identities increased the likelihood of the guessed word.

#### BERT Sentiment Analysis
Another task we can do to probe BERT's biased tendencies is to use a different sentiment analysis system and correlate overall sentiment score with amount of ability, gender, and race markers.
We will feed a basic MLM pipeline directly into sentiment analysis. We will not be using the basic BERT model for this but instead
TextBlob. This is because the vanilla sentiment analysis pipeline is a binary classifier and doesn't score text relative to each other.
TextBlob will give us a floating point rating of polarity. However, note that we have introduced another AI component, we are testing one AI with another. This is ok because we are not measuring the difference in the actual idenitity labels, but how the filled mask affects the overal score.

```python
from textblob import TextBlob

def sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

sentiment(bert()[0]['sequence'])
```

Do this and average the results across each identity set. If the overall sentiment score falls this is indicating that the MLM is predicting negative fillers.

##### Loading Data Help

```python
import pandas as pd
df = pd.read_csv('./path', delimiter='\t')
```

`df` is a [Pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

### Reflection

**q1:** At what point in the model creation process are bias mitigation techniques most effective? Why?

~~a1: In the data curation process~~

**q2:** What is the primary difficulty associated with disability biases measurement and correction?

~~a2: It is difficult to measure against a single base line due to the widely varying nature of the demographics~~

**q3:** In this lab we used MLM and sentiment analysis to measure biases. What is one other method for measuring fairness in ML systems
