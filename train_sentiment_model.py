import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# File path to your CSV (replace as necessary)
file_path = r"C:\Users\91991\Desktop\final_year_review\training.1600000.processed.noemoticon.csv"

# Load CSV file and handle structure
print("Loading dataset...")
try:
    df = pd.read_csv(
        file_path,
        encoding='ISO-8859-1',
        usecols=[0, 5],  # Column 0 = Sentiment, Column 5 = Text
        names=["Sentiment", "Text"],  # Rename columns for clarity
        header=None, 
        on_bad_lines='skip'  # Skip problematic lines
    )
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Display sample data for verification
print("Sample data:")
print(df.head())

# Map sentiment values (0 = Negative, 4 = Positive)
print("Mapping sentiment values...")
df['Sentiment'] = df['Sentiment'].replace({4: 1, 0: -1})  # Map Positive to 1, Negative to -1

# Drop rows with missing or null values in the "Text" column
print("Checking and cleaning null values...")
df.dropna(subset=["Text"], inplace=True)

# Verify dataset shape after cleaning
print(f"Dataset size after cleaning: {df.shape[0]} rows")

# Text preprocessing and vectorization
print("Vectorizing text data...")
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Limit features for efficiency
X = vectorizer.fit_transform(df["Text"])
y = df["Sentiment"]

# Train-test split
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {accuracy:.4f}")

# Save model, vectorizer, and accuracy
print("Saving model, vectorizer, and accuracy...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
with open("model_accuracy.txt", "w") as f:
    f.write(str(accuracy))

print("Model, vectorizer, and accuracy saved successfully.")
