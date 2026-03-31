# classification pipeline
# TF-IDF + LogisticRegression -> TF-IDF + Linear SVM -> Sentence embeddings 
# + classifier -> Fine-tuned BERT
# Document Clustering/Classification: Used to convert text into 
# numerical vectors for machine learning models (e.g., spam detection).

from datasets import load_dataset
import evaluate
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# 1) Load dataset
ds = load_dataset("banking77")

label_names = ds["train"].features["label"].names
num_labels = len(label_names)

# 2) Choose checkpoint
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 3) Tokenize
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,   # let data collator pad dynamically
        max_length=128,
    )

tokenized_ds = ds.map(tokenize_fn, batched=True)

# 4) Model
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels,
)

# 5) Metrics
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(
            predictions=preds,
            references=labels,
            average="macro"
        )["f1"],
    }

# 6) Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 7) Training args
training_args = TrainingArguments(
    output_dir="./bert-banking77",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9) Train + evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# 10) Inference on new text
test_query = "How do I reset my password?"
inputs = tokenizer(test_query, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
pred_label = outputs.logits.argmax(dim=-1).item()
print(label_names[pred_label])