# app.py
import streamlit as st
import joblib
import json
import pandas as pd
import re

# Model dosyasını yükle
data = joblib.load("model.pkl")
model = data["model"]
vectorizer = data["vectorizer"]
scaler = data["scaler"]
selected_indices = data["selected_indices"]
le = data["label_encoder"]
acc = data["accuracy"]
f1 = data["f1_score"]
prec = data["precision"]
rec = data["recall"]
model_name = data["model_name"]

# Başlık
st.title("🍽️🍟🍜Yemek Malzemelerine Göre Ülke Tahmini🍣🍔🍎")

# Model bilgilerini göster
st.markdown(f"""
**Kullanılan modeller:** Naive Bayes, SVM, Random Forest, KNN  
**En iyi kombinasyon:** {model_name}  

**Model Metrikleri:**
- Doğruluk: %{acc}
- F1 Score: %{f1}
- Precision: %{prec}
- Recall: %{rec}
""")

# Girdi alanı
ingredients = st.text_area("Yemeğin malzemelerini gir (örn: tomatoes, onion, soy sauce)")

if st.button("Tahmin Et"):
    if ingredients.strip() == "":
        st.warning("Lütfen malzeme girin.")
    else:
        # Ön işleme ve tahmin
        text = ingredients.lower().replace(",", " ")
        X_input = vectorizer.transform([text])
        X_scaled = scaler.transform(X_input)
        X_selected = X_scaled[:, selected_indices]

        prediction = model.predict(X_selected)[0]
        try:
            predicted_label = le.inverse_transform([int(prediction)])[0]
        except Exception as e:
            st.error(f"Tahmin etiketi çözülenirken hata oluştu:")
        else:
            st.success(f"🌍 **{predicted_label.upper()}** mutfağına ait olduğunu tahmin ediyoruz.")
            st.info(f"💡 Bu tahmin **{model_name}** kombinasyonuyla yapılmıştır.")
            st.info(f"Bir malzemenin bir mutfakta kullanılıyor olması, o yemeğin o mutfağa ait olduğunu göstermez.O yüzden model %100 doğru cevap vermeyebilir.")



        # 📊 Malzeme eşleşmeleri analizi
        st.markdown("### 📊 Malzemelerin Geçtiği Mutfaklar")
        with open("veri_seti.json", "r") as f:
            data_json = json.load(f)

        input_ingredients = [i.strip().lower() for i in re.split(r"[,\n]", ingredients) if i.strip()]
        cuisine_counts = {}

        best_match = None
        best_score = 0

        for row in data_json:
            cuisine = row["cuisine"]
            row_ingredients = set(i.lower() for i in row["ingredients"])
            match_count = sum(1 for i in input_ingredients if i in row_ingredients)

            if match_count > 0:
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + match_count
            if match_count > best_score:
                best_score = match_count
                best_match = row

        total_matches = sum(cuisine_counts.values())

        if total_matches == 0:
            st.warning("Girilen malzemeler hiçbir mutfakta bulunamadı.")
        else:
            sorted_counts = sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)
            result_df = pd.DataFrame(sorted_counts, columns=["Mutfak", "Eşleşme"])
            result_df["%"] = result_df["Eşleşme"] / total_matches * 100
            st.dataframe(result_df.style.format({"%": "{:.2f}"}))

        # 🥇 En çok eşleşen örneği göster
        if best_match:
            st.markdown("### 🥇 En Çok Eşleşen Örnek")
            st.json({
                "ID": best_match["id"],
                "Mutfak": best_match["cuisine"],
                "Malzemeler": best_match["ingredients"],
                "Eşleşen Malzeme Sayısı": best_score
            })
