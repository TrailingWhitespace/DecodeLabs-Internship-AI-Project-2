#  DecodeLabs AI Project 2: Data classification using AI

#  Architecture: IPO Model
#    INPUT  -> Load iris dataset and scale it using standard scaler
#    PROCESS -> split dataset for train and test and use KNN(K-Nearest Neighbours)
#    OUTPUT -> Produce Confusion matrix + F1 score

# Scikit-Learn
from sklearn.datasets import load_iris               # built-in Iris dataset (can also use csv files from kaggle)
from sklearn.preprocessing import StandardScaler     # scaling (more on this later)
from sklearn.model_selection import train_test_split # train/test split 
from sklearn.neighbors import KNeighborsClassifier   # the KNN algorithm
from sklearn.metrics import (                       
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# SET PLOT STYLE
# seaborn's theme makes matplotlib charts look much cleaner
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]   # blue, orange, green (one per species)



# X = features (sepal_length, sepal_width and petal_length, petal_width)
# y = labels   (0=Setosa, 1=Versicolor, 2=Virginica) (species)
iris = load_iris()
# Iris is a very famous machine learning dataset consisting of 150 values of 4 different features for 3 species of
# flowers, here iris gives us 'data' which is an Numpy array of lists where each list has 4 values for sepal/petal length and width
# 'target' here gives us the species of the flower corresponding to the values, total of 150 values
X = iris.data # list (numpy array) of lists (each list has 4 values)
y = iris.target # 150 values specifying flower species

# Shape is basically a numpy array method to find the number of items so 0 is the one corresponding to the 
# entire array which has 150 lists and 1 just picks the first list which has 4 values
print(f"\n[DATA INFO]")
print(f"  Total samples  : {X.shape[0]}") 
print(f"  Features       : {X.shape[1]} → {iris.feature_names}")
print(f"  Classes        : {iris.target_names.view()}")
samples_per_class = {
    str(name): int(sum(y == i))
    for i, name in enumerate(iris.target_names)
}
print(f"  Samples/class  : {dict(
    zip(
        map(str, iris.target_names),
        map(int, [sum(y == i) for i in range(3)])
    )
)}")

# Now the KNN algorithm works on the principle of comparing the values of an item to its K neighbours 
# to determine the type/species and it uses euclidean distance which is just the distance formula d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}
# and now we use something called StandardScaler from sklearn and the reason is
# without scaling the values which are on the very large or very small compared to the other values highly influence the
# calculations to the point where the other values do not matter and to remove this bias we use scaling
# what StandardScaler does here is convert the mean of the values to 0 and the standard deviation to 1
# Mean = just average of the values
# Standard Deviation (square root of variance) =  tells how spread out the values are


# Why/How would mean = 0 and std = 1 help?
# Example (without scaling)
x = np.array([5.1, 3.5, 1.4, 0.2]) # random set of values from the dataset
print(x.mean())  # 2.55
print(x.std())   # 1.8874586088176875
# print(x.var())   # 3.5625
print(x)
scaler_x = StandardScaler() # formula that it uses is scaled value = (value - mean) / std
# where value - mean makes 0 the mean value for the entire value set and then dividing it by std scales the values to 1
x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
print(round(x_scaled.mean())) # Mean now is essentially 0
print(np.isclose(x_scaled.mean(), 0)) 
print(x_scaled.std()) # Std is 1 now 
# Std is now 1, meaning the data has been scaled so that
# one unit represents one standard deviation from the mean
# the other way i understand this is that we have lengths ranging from 3-7 or 9 and the y or labels here are only 0,1,2 
# this means that the lengths would dominate 
# but by scaling x, both set of values x and y turn into nearly 1 or 2 or 0 which aligns with both the data  
# so in simple terms we are not letting the larger data make the predictions biased where here x is larger so we are only scaling it
print(x_scaled.flatten())  

# So std before scaling is 1.88 and after 1 and the difference here is not much because its the iris dataset
# any other dataset the difference would be more significant

# Now for mean = 0 it helps that ml algorithms care about distance from the average and an example for that would be
# Original:
# [100, 110, 120]
# Mean = 110
# Subtract the mean:
# [-10, 0, 10]
# Now:
# Mean = 0

# So if 110 is 0, 100 is -10 away and -10 is what corresponds to 100 in the new list
# same with 120 and 10
# so the absolute values are being changed but not the distance between those values
# essentially the same but now the mean is 0 so its easier to calculate how far the other values are from the average
# so a value 5.9 would mean that it is 5.9 standard deviations above  the mean (0)


print(f"\n[SCALING]")
# Now to scale our X ('data') values
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
print(f"{"Before scaling : ":>20}", end = "  ")
print([float(val) for val in X[0]], end = "        ")
print([float(val) for val in X[1]])
print(f"{"After scaling : ":>20}", end = "  ")
print(f"{str([float(val) for val in X_scaled.round(2)[0]])}", end="  ")
print(f"{str([float(val) for val in X_scaled.round(2)[1]])}")
# This function does 2 operations at once 
# here fit calculates the mean and std for all the value sets so all 150 of them
# transform uses the value - mean / std formula on each value to scale it 



# TODO: Part about fitting train and test data separately VS only fitting train data
# X_scaled = scaler.fit_transform(X)
# train_test_split(X_scaled, y, ...)
 
# TO

# train_test_split(...)
# fit scaler on X_train
# transform X_train
# transform X_test



# Now for splitting the data into 2 parts, one for training and the other for testing later
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20% test, 80% train
    random_state=42,    # reproducible shuffle
    stratify=y          # balanced class distribution in both splits
)
# here test size 0.2 means 20% of 150 which is 30 is saved for testing 
# random state is used because otherwise the split is completely random each run which produces different results 
# so the arbitrary number 42 acts like a seed 
# stratify = y here makes sure the classes are proportionate between the train and test data set, 
# train_test_split shuffles the data by default so the classes (y labels) are also shuffled by they might not be proportionate 
# which could give us something like
# Train:
#   Setosa      42
#   Versicolor  39
#   Virginica   39

# Test:
#   Setosa       8
#   Versicolor  11
#   Virginica   11
# while stratify converts it to
# Setosa:
#   40 train
#   10 test
# Versicolor:
#   40 train
#   10 test
# Virginica:
#   40 train
#   10 test

# Result:
# Train:
#   Setosa      40
#   Versicolor  40
#   Virginica   40
# Test:
#   Setosa      10
#   Versicolor  10
#   Virginica   10

print(f"\n[SPLIT]")
print(f"  Training samples : {len(X_train)} (80%)")
print(f"  Testing samples  : {len(X_test)}  (20%)")

# KNN Algorithm
# How it works:
#   1. For a value, measure the distance to all other values (euclidean distance formula)
#   2. Find the K closest neighbors
#   3. Take a majority vote among those K neighbors
#   4. Assign the majority class as the prediction
#
# Choosing K (using elbow)
# K=1   → overfits (only 1 comparison which makes it really sensitive to whatever that is)
# K=100 → underfits (too many neighbors, prediction becomes meaningless)
# K=5  → a reliable starting point for small datasets like Iris

K = 5 

model = KNeighborsClassifier(n_neighbors=K)

# FIT (store all training points + their labels)
# KNN does NO computation at training time
# All the work happens at prediction time
model.fit(X_train, y_train)

predictions = model.predict(X_test)


print(f"\n[PREDICTIONS vs ACTUAL]")
print(f"{'#':<5} {'Predicted':<15} {'Actual':<15} {'Correct?'}")
print("-" * 45)
for i, (pred, actual) in enumerate(zip(predictions, y_test)):
    pred_name   = iris.target_names[pred]
    actual_name = iris.target_names[actual]
    correct     = "✅" if pred == actual else "❌"
    print(f"{i:<5} {pred_name:<15} {actual_name:<15} {correct}")

# manual accuracy 
print(f"\n[ACCURACY]")
correct_count = sum(1 for p, a in zip(predictions, y_test) if p == a)
total         = len(y_test)
accuracy      = correct_count / total * 100
sklearn_accuracy = accuracy_score(y_test, predictions) * 100
print(f"  Algorithm  : K-Nearest Neighbors")
print(f"  K value    : {K}")
print(f"  accuracy_score() : {sklearn_accuracy:.1f}%")
print(f"  Manual calc      : {accuracy:.1f}%")
print(f"  {correct_count} correct out of {total} flowers = {accuracy:.1f}%")

# errors
print(f"\n[ERRORS]")
for class_index, species in enumerate(iris.target_names):
    actual_this_class = [i for i, a in enumerate(y_test) if a == class_index]
    wrong = [i for i in actual_this_class if predictions[i] != y_test[i]]
    right = len(actual_this_class) - len(wrong)
    print(f"  {species:<12} : {right}/{len(actual_this_class)} correct", end="")
    for i in wrong:
        print(f"  ← #{i} called '{iris.target_names[predictions[i]]}'", end="")
    print()


# Best way to understand accuracy mirage, confusion matrix and f1 score
# Accuracy mirage 
# accuracy is a lie/not useful for an imbalanced dataset 
# example
# Example: Imbalanced dataset
# detecting a rare disease:
# Healthy: 990 people
# Disease: 10 people
# Results:
# Correct predictions = 990 healthy
# Total predictions   = 1000
# Accuracy = 990 / 1000 = 99%
# it failed at predicting the 10 sick people

# Confusion matrix
# Rows = actual classes, Columns = predicted classes
# Diagonal = correct predictions (TP for each class)
# Off-diagonal = errors (FP / FN)
# TP | FP 
# FN | TN
# true positive (TP) : the prediction was true for something true
# false positive (FP) : the prediction was true for something false
# false negative (FN) : the prediction was false for something true
# true negative (TN) : the prediction was false for something false

# F1 SCORE

# Precision
# "Out of everyone the model flagged as positive, how many actually were?"
# Model screened 1000 emails and marked 100 as spam
# Of those 100, only 70 were actually spam
# 30 were real emails wrongly deleted
# Precision = 70/100 = 70%
# Precision = trustworthiness of positive predictions. When the model says yes, how often is it right?
# Low precision = lots of false alarms. In a spam filter this means real emails getting deleted. Annoying.

# Recall
# "Out of all the actual positives, how many did the model catch?"
# There were 100 actual spam emails in total
# The model caught 70 of them
# 30 spam emails slipped through to the inbox
# Recall = 70/100 = 70%
# Recall = how many real cases did you find. When something is actually positive, how often does the model catch it?
# Low recall = lots of missed detections. In cancer screening this means sick patients being told they're healthy. Dangerous.

# The trade-off
# improving one usually hurts the other
# SPAM FILTER example:
# Be very aggressive (flag everything suspicious):
# → Catch almost all spam         → Recall goes UP
# → Also delete lots of real mail → Precision goes DOWN

# Be very conservative (only flag obvious spam):
# → Almost never delete real mail → Precision goes UP
# → Miss lots of actual spam      → Recall goes DOWN

# F1 Score — balancing both
# F1 is the harmonic mean of precision and recall:
# F1 = 2 × (Precision × Recall) / (Precision + Recall)
# Why harmonic mean and not regular average? Because regular average can be fooled:
# Precision = 100%,  Recall = 0%

# Regular average = (100 + 0) / 2 = 50%  ← sounds okay?

# Harmonic mean  = 2×(100×0)/(100+0) = 0%  ← correctly shows this is useless
# The harmonic mean punishes extreme imbalance between precision and recall
# A model scoring 0% on either gets an F1 close to 0, no matter how high the other one is


cm = confusion_matrix(y_test, predictions)
print(f"\n[CONFUSION MATRIX]")
print(f"  Rows = Actual | Columns = Predicted")
print(f"  Classes: {[str(name) for name in iris.target_names]}\n")

# Print the matrix with labels for clarity
header = f"{'':22}" + "  ".join(f"{name:12}" for name in iris.target_names)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{val:12}" for val in row)
    print(f"  {iris.target_names[i]:12}{row_str}")

f1 = f1_score(y_test, predictions, average='weighted')
print(f"\n[F1 SCORE]  {f1:.4f}  (1.0 = perfect, 0.0 = useless)")

# Full per-class breakdown: precision, recall, f1, support
print(f"\n[FULL CLASSIFICATION REPORT]")
print(classification_report(y_test, predictions, target_names=iris.target_names))


# Visuualizations for better understanding

# ============================================================
#  CHART 1 — CONFUSION MATRIX HEATMAP
#  What it shows: exactly where the model is right/wrong
#  per class. Darker square = more predictions in that cell.
#  Perfect model = dark diagonal, white everywhere else.
# ============================================================

def plot_confusion_matrix(y_test, predictions, names):
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(7, 6))

    # annot=True prints the number inside each cell
    # fmt='d' means integer format
    # cmap='Blues' gives the color gradient
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=names,
        yticklabels=names,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"}
    )

    ax.set_xlabel("Predicted Species", fontsize=13, labelpad=10)
    ax.set_ylabel("Actual Species",    fontsize=13, labelpad=10)
    ax.set_title("Confusion Matrix\nDark diagonal = correct | Off-diagonal = mistakes",
                 fontsize=13, pad=15)

    # Add a text explanation below the chart
    total   = cm.sum()
    correct = cm.trace()        # .trace() = sum of diagonal
    fig.text(0.5, 0.01,
             f"Total: {total} flowers  |  Correct: {correct}  |  Wrong: {total-correct}",
             ha='center', fontsize=11, color='gray')

    plt.tight_layout()
    plt.savefig("chart1_confusion_matrix.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Chart 1 saved — Confusion Matrix")

# ============================================================
#  CHART 2 — K vs ERROR RATE (THE ELBOW CURVE)
#  What it shows: how model performance changes as K changes.
#  Recreates the elbow diagram from slide 12 with real data.
#  Low K = overfitting (high error), High K = underfitting.
#  The sweet spot is the elbow — lowest error point.
# ============================================================

def plot_k_vs_error(X_train, y_train, X_test, y_test):
    k_range      = range(1, 31)   # test every K from 1 to 30
    error_rates  = []
    accuracy_rates = []

    for k in k_range:
        knn  = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, preds))   # error = 1 - accuracy
        accuracy_rates.append(accuracy_score(y_test, preds))

    best_k     = k_range[error_rates.index(min(error_rates))]
    best_error = min(error_rates)
    best_accuracy = max(accuracy_rates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Error Rate ──
    ax1.plot(k_range, error_rates,
             color='#F44336', marker='o', markersize=6,
             linewidth=2, label='Error Rate')
    ax1.axvline(x=best_k, color='gray', linestyle='--', linewidth=1.5,
                label=f'Best K = {best_k}')
    ax1.scatter([best_k], [best_error],
                color='gold', s=200, zorder=5, edgecolors='black',
                label=f'Elbow (error={best_error:.2f})')
    ax1.set_xlabel("K Value",     fontsize=12)
    ax1.set_ylabel("Error Rate",  fontsize=12)
    ax1.set_title("K vs Error Rate\nLow K=overfit | High K=underfit | Elbow=sweet spot",
                  fontsize=12)
    ax1.legend(fontsize=10)

    # ── Right: Accuracy ──
    ax2.plot(k_range, accuracy_rates,
             color='#2196F3', marker='o', markersize=6,
             linewidth=2, label='Accuracy')
    ax2.axvline(x=best_k, color='gray', linestyle='--', linewidth=1.5,
                label=f'Best K = {best_k}')
    ax2.scatter([best_k], [best_accuracy],
    color='gold', s=200, zorder=5, edgecolors='black',
    label=f'Elbow (Accuracy={best_accuracy:.2f})')
    ax2.set_xlabel("K Value",    fontsize=12)
    ax2.set_ylabel("Accuracy",   fontsize=12)
    ax2.set_title("K vs Accuracy\n(mirror of error rate)",
                  fontsize=12)
    ax2.legend(fontsize=10)

    plt.suptitle("Tuning the Engine: Choosing K",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("chart3_k_vs_error.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Chart 2 saved — K vs Error Rate")

# ============================================================
#  CHART 3 — PRECISION, RECALL & F1 PER CLASS (BAR CHART)
#  What it shows: the three key metrics side by side for
#  each species. Instantly shows which class the model
#  is weakest on and whether the weakness is in precision
#  or recall — which tells you what TYPE of errors it makes.
# ============================================================

def plot_metrics_per_class(y_test, predictions, names):
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, predictions, average=None)
    recall    = recall_score(y_test,    predictions, average=None)
    f1        = f1_score(y_test,        predictions, average=None)

    x     = np.arange(len(names))   # [0, 1, 2] positions on x axis
    width = 0.25                    # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width,
                   label='Precision', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x,          recall,    width,
                   label='Recall',    color='#FF9800', alpha=0.85)
    bars3 = ax.bar(x + width,  f1,        width,
                   label='F1 Score',  color='#4CAF50', alpha=0.85)

    # Add value labels on top of each bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score (0 = worst, 1 = best)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision, Recall & F1 Score per Species\n"
                 "Precision = trustworthiness | Recall = coverage | F1 = balance of both",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig("chart5_metrics_per_class.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Chart 3 saved — Precision / Recall / F1 per class")


# ============================================================
#  CHART 4 — OVERALL SCORECARD (SUMMARY DASHBOARD)
#  What it shows: a single clean summary of all key numbers
#  in one place. Good to include as a final slide/summary.
# ============================================================

def plot_scorecard(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    f1  = f1_score(y_test, predictions, average='weighted')
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, predictions, average='weighted')
    rec  = recall_score(y_test,    predictions, average='weighted')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values  = [acc, prec, rec, f1]
    colors_bar = ['#9C27B0', '#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(metrics, values, color=colors_bar,
                   alpha=0.85, edgecolor='white', height=0.5)

    # Value labels inside/outside bars
    for bar, val in zip(bars, values):
        ax.text(val - 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}',
                va='center', ha='right',
                fontsize=14, fontweight='bold', color='white')

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Overall Model Scorecard — KNN (K=5) on Iris Dataset",
                 fontsize=13, pad=15)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1)

    # Add a note about accuracy mirage
    fig.text(0.5, 0.01,
             "Note: Iris is balanced (50/class) so accuracy is reliable here. "
             "On imbalanced data, always prefer F1.",
             ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig("chart6_scorecard.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅  Chart 4 saved — Overall Scorecard")


names     = iris.target_names        # ['setosa', 'versicolor', 'virginica']
features  = iris.feature_names       # the 4 measurement column names

print("\n" + "="*45)
print("  Generating all visualizations...")
print("="*45 + "\n")
plot_confusion_matrix(y_test, predictions, names)
plot_k_vs_error(X_train, y_train, X_test, y_test)
plot_metrics_per_class(y_test, predictions, names)
plot_scorecard(y_test, predictions)

# TODO: Understand how predicting new flowers works with why we use the same scaler and single transform fucntion and predict_proba etc.
