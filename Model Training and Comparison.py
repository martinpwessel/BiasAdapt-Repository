import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, RandomSampler,random_split,SubsetRandomSampler
from sklearn.metrics import f1_score
from transformers import EarlyStoppingCallback
import random
from collections import Counter
import matplotlib.pyplot as plt

class DataLoad:
    """
    Class to read in the BABE Dataset
    """
    @staticmethod
    def read_babe():
        df = pd.read_excel("../input/babe-media-bias-annotations-by-experts/data/final_labels_SG2.xlsx")
        lst = []
        for index, row in df.iterrows():
            if row['label_bias'] == "No agreement":
                pass
            else:
                sub_dict = {'text': row['text']}
                if row['label_bias'] == "Biased":
                    sub_dict['label'] = 1
                elif row['label_bias'] == "Non-biased":
                    sub_dict['label'] = 0
                lst.append(sub_dict)
        return lst

data = DataLoad.read_babe()



# Function to calculate word count for a given text
def word_count(text):
    return len(text.split())


# Counting word occurrences in the dataset
word_counts = Counter()

for item in data:
    text = item['text']
    count = word_count(text)
    word_counts[count] += 1

# Plotting the word count distribution
counts, frequencies = zip(*word_counts.items())

plt.bar(counts, frequencies, color='blue')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Distribution')
plt.show()

df = pd.read_excel('/kaggle/input/augmented-adverserial-data-set/combined_output.xlsx')
# df = df.drop(columns=['original'])
data_augmented = df.to_dict(orient='records')


def filtering(data_augmented):
    filtered_dataset = []

    i = 0
    while i < len(data_augmented):
        entry = data_augmented[i]

        if entry['original'] == 1:
            filtered_dataset.append(entry)
            # Check if there is an augmented entry
            if i + 1 < len(data_augmented) and data_augmented[i + 1]['original'] != 1:
                filtered_dataset.append(data_augmented[i + 1])
            if i + 2 < len(data_augmented) and data_augmented[i + 2]['original'] != 1:
                filtered_dataset.append(data_augmented[i + 2])

        i += 1  # Move to the next set of entries
    return filtered_dataset


data_filtered = filtering(data_augmented)


def aug_train_test_split(data_augmented):
    train_dataset = []
    test_dataset = []

    i = 0
    while i < len(data_augmented):
        entry = data_augmented[i]

        if entry['original'] == 1:
            if random.random() < 0.8:  # 80% chance to go to the train dataset
                train_dataset.append(entry)
                # Check if there is an augmented entry
                y = 1
                x = 1
                while i + y < len(data_augmented) and data_augmented[i + y]['original'] != 1:
                    train_dataset.append(data_augmented[i + y])
                    x = 1 + y
                    y += 1

            else:  # 20% chance to go to the test dataset
                test_dataset.append(entry)
                # Check if there is an augmented entry
                y = 1
                x = 1
                while i + y < len(data_augmented) and data_augmented[i + y]['original'] != 1:
                    test_dataset.append(data_augmented[i + 1])
                    x = 1 + y
                    y += 1

        i += x  # Move to the next original entry, skipping the augmented one
    return train_dataset, test_dataset


aug_train_dataset, aug_test_dataset = aug_train_test_split(data_filtered)

# Load spurious cues test set

with open('/kaggle/input/spurious-cues-testset/spurious_cues_test', 'rb') as file:
    sc_testset = pickle.load(file)

def compute_metrics(p):
    pred_flat = np.argmax(p.predictions, axis=1).flatten()
    labels_flat = p.label_ids.flatten()
    return {'f1': f1_score(labels_flat, pred_flat)}


def tokenizing_function(data) -> list:
    model_name = "mediabiasgroup/magpie-babe-ft"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenizing
    tokenized = []
    for i in range(len(data)):
        token = tokenizer(data[i]["text"], padding="max_length", truncation=True, max_length=75)
        token['labels'] = data[i]['label']
        tokenized.append(token)
    ten = []
    for i in range(len(tokenized)):
        x = {}
        for j in tokenized[i].keys():
            x[j] = torch.tensor(tokenized[i][j])
        ten.append(x)
    return ten

def training(train_data, val_data):
    # Tokenize your data
    train_tokenized = tokenizing_function(train_data)
    val_tokenized = tokenizing_function(val_data)

    # Initialize DataLoader
    train_dataloader = DataLoader(train_tokenized, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_tokenized, batch_size=16, shuffle=False)

    # Initialize the RoBERTa model and the tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("mediabiasgroup/magpie-babe-ft", num_labels=2)
    #model_name = "mediabiasgroup/lbm_without_media_bias_pretrained"
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Arguments with aligned save and evaluation strategies
    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,  # Set according to your needs
        learning_rate=2e-5,
        logging_dir="./logs",
        output_dir="./finetuned_model",
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",  # Save after each epoch to match the evaluation strategy
        save_total_limit=5,  # Only keep the last 5 models
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Use eval_loss for early stopping
        greater_is_better=False,  # Smaller eval_loss is better
    )

    # Initialize Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Setting early stopping patience
    )

    # Train the model
    trainer.train()

    return model


aug_model = training(aug_train_dataset, aug_test_dataset)


def evaluate_spurious_cues(sc_testset, model, tokenizer):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    categories_data = {
        'Gender': ['Male', 'Female', 'Non-binary'],
        'Origin': ['European', 'African', 'Asian'],
        'Religion': ['Christian', 'Islam', 'Atheist'],
        'Politician': ['Conservatives', 'Liberals', 'Socialists'],
        'Political Affiliation': ['Left-wing (liberal/progressive)', 'Right-wing (conservative)',
                                  'Centrist (Moderate)'],
        'Occupation': ['Services', 'Creative Arts and Media', 'Skilled Trades and Manual Labour']
    }

    summary_table = []

    for i in sc_testset:
        for key in i.keys():
            bias = key
        final_sentences = i[key]
        model_predictions = []

        for sentence, ground_truth, category in final_sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Make a prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu()
            predicted_label = np.argmax(logits, axis=1).item()
            model_predictions.append((sentence, predicted_label, ground_truth, category))

        # Store metrics in a structured format
        metrics_list = []
        f1_scores = []
        for category in categories_data[bias]:
            ground_truth_labels = [x[2] for x in model_predictions if x[3] == category]
            predicted_labels = [x[1] for x in model_predictions if x[3] == category]

            if ground_truth_labels:  # To ensure categories with no examples don't break the analysis
                accuracy = accuracy_score(ground_truth_labels, predicted_labels)
                precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
                recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
                f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)
                metrics_list.append([category, accuracy, precision, recall, f1])
                f1_scores.append(f1)

        # Convert to DataFrame for better display
        df_metrics = pd.DataFrame(metrics_list, columns=["Category", "Accuracy", "Precision", "Recall", "F1 Score"])
        print(f"\nEvaluation Metrics for Bias Type: {bias}")
        print(df_metrics)

        # Plot grouped bar chart for F1 Scores
        plt.figure(figsize=(10, 6))
        df_metrics.plot(kind='bar', x='Category', y=['F1 Score'], color='skyblue', legend=False)
        plt.title(f"F1 Score Comparison for {bias}")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Compute average and variance of F1 scores
        avg_f1 = np.mean(f1_scores)
        var_f1 = np.var(f1_scores)
        summary_table.append([bias, avg_f1, var_f1] + f1_scores)

        # Collect overall results
        avg_metrics = df_metrics[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
        print(f"\nAverage Metrics for Bias Type '{bias}':\n{avg_metrics}")
        results.append({"bias": bias, "metrics": df_metrics, "average_metrics": avg_metrics.to_dict()})

    # Create the summary table DataFrame
    summary_columns = ['Category', 'Average F1 Score', 'Variance'] + [f'F1_{cat}' for cat in categories_data]
    summary_df = pd.DataFrame(summary_table, columns=summary_columns[:len(summary_table[0])])

    print("\nSummary Table (F1 Scores, Averages, Variances):")
    print(summary_df)

    return results, summary_df


tokenizer = AutoTokenizer.from_pretrained("mediabiasgroup/magpie-babe-ft")
model = AutoModelForSequenceClassification.from_pretrained("mediabiasgroup/magpie-babe-ft", num_labels=2)
results_aug, summary_aug = evaluate_spurious_cues(sc_testset, aug_model, tokenizer)

results_model, summary_model = evaluate_spurious_cues(sc_testset, model, tokenizer)
print(summary_aug, summary_model)
