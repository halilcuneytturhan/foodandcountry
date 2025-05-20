                                #KÜTÜPHANELER
import matplotlib
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


#Veri setini kaça kaç böldüğümüzü gösteren grafik
def save_data_split_pie_chart(train_size, test_size):
    sizes = [train_size, test_size]
    labels = ['Eğitim Verisi (%80)', 'Test Verisi (%20)']
    colors = ['#00008b', '#ff0000']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Veri Seti Dağılımı (Toplam %100)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("veri_dagilimi.png")
    plt.close()
    
    
    #MODEL KARŞILAŞTIRILMASI
def save_metric_comparison_plots(results_df):
    metrics = ["Accuracy(Doğruluk)", "F1-Score", "Precision(Kesinlik)", "Recall(Duyarlılık)"]
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Modeller", y=metric, hue="Özellik seçimi", data=results_df, palette="Set2")
        plt.title(f"Model Başarı Karşılaştırması - {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{metric.lower()}_ws.png")
        plt.close()

def main():

    # veri okuma
    df = pd.read_json("train.json")
    df["text"] = df["ingredients"].apply(lambda ing: re.sub(r"[^a-zA-Z\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc\s]", "", " ".join(ing).lower()))

    # temiz veri 
    X_raw = df["text"]
    y = df["cuisine"]

    # label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X_raw)

    # MaxAbsScaler
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_vect)

    # Özellik seçimi yöntemleri
    feature_selectors = {
        "SelectKBest": SelectKBest(score_func=chi2, k=1000),
        "RFE": RFE(estimator=LinearSVC(max_iter=5000), n_features_to_select=1000, step=0.1),
        "Tree-Based": SelectKBest(score_func=lambda X, y: RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y).feature_importances_, k=1000)
    }

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    best_acc = 0
    best_model = None
    best_name = ""
    best_selector_name = ""
    best_metrics = {}
    results = []
    #model eğitme
    for sel_name, selector in feature_selectors.items():
        print(f"\n--- Özellik Seçimi: {sel_name} ---")
        X_selected = selector.fit_transform(X_scaled, y_encoded)
        #%80 eğitme, %20 test olarak ayırıyor.
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        if sel_name == "SelectKBest":
            save_data_split_pie_chart(X_train.shape[0], X_test.shape[0])

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)

            print(f"{model_name} - Doğruluk: %{acc*100:.2f}, F1: %{f1*100:.2f}, Precision: %{prec*100:.2f}, Recall: %{rec*100:.2f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
            plt.title(f"Confusion Matrix: {sel_name} + {model_name}")
            plt.tight_layout()
            filename = f"confusion_matrix_{sel_name}_{model_name}.png".replace(" ", "_")
            plt.savefig(filename)
            plt.close()

            results.append({
                "Modeller": model_name,
                "Özellik seçimi": sel_name,
                "Accuracy(Doğruluk)": acc,
                "F1-Score": f1,
                "Precision(Kesinlik)": prec,
                "Recall(Duyarlılık)": rec
            })

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = model_name
                best_selector_name = sel_name
                best_metrics = {'f1': f1, 'precision': prec, 'recall': rec}

    results_df = pd.DataFrame(results)
    save_metric_comparison_plots(results_df)

    joblib.dump({
        "model": best_model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "selected_indices": feature_selectors[best_selector_name].get_support(indices=True) if best_selector_name != "Tree-Based" else
            np.argsort(feature_selectors["Tree-Based"].score_func(X_scaled, y_encoded))[::-1][:1000],
        "label_encoder": le,
        "accuracy": round(best_acc * 100, 2),
        "f1_score": round(best_metrics['f1'] * 100, 2),
        "precision": round(best_metrics['precision'] * 100, 2),
        "recall": round(best_metrics['recall'] * 100, 2),
        "model_name": f"{best_selector_name} + {best_name}"
    }, "model.pkl")

    print(f"\n✅ En iyi model: {best_selector_name} + {best_name}")
    print(f"Doğruluk: %{best_acc*100:.2f}")
    print(f"F1 Score: %{best_metrics['f1']*100:.2f}")
    print(f"Precision: %{best_metrics['precision']*100:.2f}")
    print(f"Recall: %{best_metrics['recall']*100:.2f}")



if __name__ == "__main__":
    main()
