# README

The purpose of these two models are to answer a question about a given text when prompted. Additionally, the model should be able to refrain from answering a question that is unanswerable given the provided context.

We use the first nine topics from [SQuAD v2.0]([SQuAD - the Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/))'s dataset to benchmark our two models, which contains around three thousand questions with an even split between answerable and unanswerable. This truncated version can be found in this repository.

### Building the prediction json:

Dependencies for both models:

1. transformers

2. pyTorch

3. tensorflow

These can be installed using pip.

```bash
pip install <name>
```

Both questionAnsweringBERT.py and questionAsnweringT5.py are used to construct their respective prediction dictionary for the questions in the first nine topics of the SQuAD dataset. Run the following commands to output the two json which can then be used as input for our benchmarks.

```bash
python3 questionAnsweringBERT.py
```

```bash
python3 questionAsnweringT5.py
```



### Benchmarking:

Beginning with our human heuistic, simply run the following command to begin the command line tool.

```bash
python3 humanEval.py
```

The user will be prompted with a context, a question, the ground truth answer, and the predictions from both models (which are presented in a random order each time). Inputing a 1 or a 2 will indicate preference for one prediction or the other while inputing a 0 will award neither a point. A continous score will be kept for the users preference towards each model, which will be displayed each time the user inputs a preference.

Secondly, to run our word vector metric, run the following command.

```python3
python3 naiveWordVecMetric.py
```

**Note** that the file name is hard-coded as BERT1-9.json, so if you would like to test the FLAN-T5 model, that will need to be swapped in the code to T51-9.json. This may also take a moment to run since its downloading gensim's "glove-wiki-gigaword-50" word vector embeddings.

Finally, to run the SQuAD evaluation script for each of our predicted results, we can run the following command.

```bash
python3 squadEvaluationScript.py dev-v2.0.json <BERT or T5 json>
```
