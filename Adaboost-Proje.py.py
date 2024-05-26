import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Verilerin Import Edilmesi
dataset = pd.read_csv(r"breast_cancer.csv") # csv uzantılı veri setinin dosya konumunu yazmanız gerekli. 

# Train ve Test Setleri
def split_data(X, y, test_size=0.2, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state) # Veri kümesini eğitim ve test setlerine ayırma

# Modelin Kurulması ve Eğitilmesi
def train_model(X_train, y_train, n_estimators=100):
    classifier = AdaBoostClassifier(n_estimators=n_estimators, random_state=0, algorithm='SAMME') #Kullanılan eğitim metodu
    classifier.fit(X_train, y_train) # Modeli eğitim metodu ile eğitme
    return classifier

# Modelin Değerlendirilmesi
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)  # Test seti üzerinde tahmin yapma
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) # Doğruluk hesaplama
    precision = precision_score(y_test, y_pred, pos_label=2) # Hassasiyet hesaplama
    recall = recall_score(y_test, y_pred, pos_label=2) #Geri çağırma skorunu hesaplama
    f1 = f1_score(y_test, y_pred, pos_label=2) # F1 skorunu hesaplama
    return cm, accuracy, precision, recall, f1


# Metriklerin Grafiğinin Oluşturulması
def plot_metrics(precision, recall, f1):
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=['yellow', 'orange', 'brown'])
    plt.title("Model Performance Metrics\n(Modelin Performans Ölçümleri)")
    plt.xlabel("Metrics\n(Ölçümler)")
    plt.ylabel("Values\n(Değerler)")
    plt.show()

# Confusion Matrix'in Grafiğinin Oluşturulması
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix\n(Karışıklık Matrisi)")
    plt.xlabel("Predicted Labels\n(Tahmin Edilen Etiketler)")
    plt.ylabel("True Labels\n(Doğru Etiketler)")
    plt.show()

# k-Fold Cross Validation ile Modelin Performansının Ölçülmesi
def cross_validation(X, y, classifier, cv=10):
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=cv)
    mean_accuracy = accuracies.mean() * 100
    std_deviation = accuracies.std() * 100
    return mean_accuracy, std_deviation

# Veri Hazırlığı
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = split_data(X, y)

# Modeli Eğitme
classifier = train_model(X_train, y_train)

# Modeli Değerlendirme
cm, accuracy, precision, recall, f1 = evaluate_model(classifier, X_test, y_test)

# Sonuçların Çıktısını Alma
plot_metrics(precision, recall, f1)
plot_confusion_matrix(cm)
print(f"Precision(Hassasiyet): {precision}\nRecall(Geri Çağırma): {recall}")
print("F1 Score:", f1)


# k-Fold Cross Validation ile Modelin Performansını Ölçme
mean_accuracy, standart_sapma= cross_validation(X, y, classifier)
print("Mean Accuracy (Ortalama Doğruluk):", mean_accuracy)
