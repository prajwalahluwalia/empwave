# Empwave training dataset

Generate the dataset from the locally downloaded sources:

```bash
python3 scripts/build_dataset.py --source-root ~/Desktop/data
```

Outputs are written to `data/processed/{train,validation,test}.jsonl` with
`label_schema.json` and `dataset_stats.json`.

GoEmotions labels remain as their original human-annotated multi-label
emotions. MultiNLI has no emotion annotations, so balanced hypotheses receive
separate semantic intent labels with stricter thresholds and a lower
recommended training weight (`0.45`). Every record retains its source and
labeling method. Brain regions are derived later from emotion and intent
evidence rather than used as the NLP training target.

The version-2 training script currently consumes the GoEmotions records.
MultiNLI weak intent records are retained for intent-layer experimentation and
evaluation; runtime semantic intents currently use the reviewed prototype
concepts in `empwave/services/intent_classifier.py`.

This is an illustrative cognitive dataset, not measured neuroimaging ground
truth. Review source licenses, dataset cards, biases, and potentially offensive
content before redistributing or training a public model.

## Train the classifier

After generating the processed files, run:

```bash
python3 scripts/train_model.py
```

This trains one supervised logistic classifier per GoEmotions label over
frozen `all-MiniLM-L6-v2` sentence embeddings. It tunes decision thresholds on
the validation split and reports final multi-label emotion metrics on the test
split. Outputs are:

- `models/trained/empwave_classifier.joblib`
- `models/trained/training_metrics.json`

The dataset has very few positive examples for some anatomical regions, so
their individual metrics are expected to be unstable until more reviewed
training examples are added.
