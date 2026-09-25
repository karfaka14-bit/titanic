# ==========================================================
# TITANIC SURVIVAL PREDICTION PROJECT
# Machine Learning + Data Visualization
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================================
# 1. قراءة بيانات Titanic
# ==========================================================

df = pd.read_csv("Titanic-Dataset.csv")

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)


# ==========================================================
# 2. تحديد البيانات التي سنستخدمها
# ==========================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"


X = df[features]
y = df[target]


# ==========================================================
# 3. تقسيم البيانات إلى Training و Testing
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 4. تحديد الأعمدة الرقمية والنصية
# ==========================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# ==========================================================
# 5. معالجة البيانات الرقمية
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ==========================================================
# 6. معالجة البيانات النصية
# ==========================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ==========================================================
# 7. دمج معالجة البيانات
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================================
# 8. إنشاء نموذج Machine Learning
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==========================================================
# 9. إنشاء Pipeline
# ==========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ==========================================================
# 10. تدريب النموذج
# ==========================================================

print("\nTraining the model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ==========================================================
# 11. اختبار النموذج
# ==========================================================

y_pred = pipeline.predict(X_test)


# ==========================================================
# 12. حساب Accuracy
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================================
# 13. Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# 14. التنبؤ براكب جديد
# ==========================================================

new_passenger = pd.DataFrame(
    {
        "Pclass": [6],
        "Sex": ["male"],
        "Age": [29],
        "SibSp": [0],
        "Parch": [0],
        "Fare": [100],
        "Embarked": ["S"]
    }
)


prediction = pipeline.predict(
    new_passenger
)

probability = pipeline.predict_proba(
    new_passenger
)


# ==========================================================
# 15. عرض نتيجة التنبؤ
# ==========================================================

print("\n" + "=" * 60)
print("NEW PASSENGER PREDICTION")
print("=" * 60)

print("\nPassenger Information:")
print(new_passenger)


if prediction[0] == 1:

    print("\nPrediction: SURVIVED")

else:

    print("\nPrediction: DID NOT SURVIVE")


print(
    f"Survival Probability: "
    f"{probability[0][1] * 100:.2f}%"
)

print(
    f"Death Probability: "
    f"{probability[0][0] * 100:.2f}%"
)


# ==========================================================
# 16. الرسم الأول:
# عدد الناجين وغير الناجين
# ==========================================================

survival_counts = df["Survived"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    ["Did Not Survive", "Survived"],
    [
        survival_counts.get(0, 0),
        survival_counts.get(1, 0)
    ]
)

plt.title(
    "Titanic Survival Distribution"
)

plt.xlabel(
    "Survival Status"
)

plt.ylabel(
    "Number of Passengers"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "survival_distribution.png",
    dpi=300
)

plt.show()


# ==========================================================
# 17. الرسم الثاني:
# العلاقة بين العمر والنجاة
# ==========================================================

plt.figure(figsize=(8, 5))

survived_data = df[
    df["Survived"] == 1
]

not_survived_data = df[
    df["Survived"] == 0
]


plt.scatter(
    survived_data["Age"],
    survived_data["Fare"],
    label="Survived",
    alpha=0.6
)

plt.scatter(
    not_survived_data["Age"],
    not_survived_data["Fare"],
    label="Did Not Survive",
    alpha=0.6
)

plt.title(
    "Age and Fare vs Survival"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Fare"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "age_fare_survival.png",
    dpi=300
)

plt.show()


# ==========================================================
# 18. الرسم الثالث:
# النجاة حسب الجنس
# ==========================================================

survival_by_sex = pd.crosstab(
    df["Sex"],
    df["Survived"]
)

plt.figure(figsize=(8, 5))

survival_by_sex.plot(
    kind="bar"
)

plt.title(
    "Titanic Survival by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Number of Passengers"
)

plt.xticks(
    rotation=0
)

plt.legend(
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

plt.savefig(
    "survival_by_gender.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)