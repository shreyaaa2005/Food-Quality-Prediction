from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ✅ Correct model path — make sure file exists here
model = tf.keras.models.load_model("models/food_quality_model.h5")

# Class labels
class_names = ['Healthy', 'Unhealthy', 'Spoiled']

# Folder for uploads
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # --- Image preprocessing ---
            img = Image.open(file_path).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            # --- Prediction ---
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]

            # --- Portion size estimation ---
            width, height = img.size
            area = width * height
            if area < 250000:
                portion = "Small"
            elif area < 600000:
                portion = "Medium"
            else:
                portion = "Large"

            # --- Nutrition estimation ---
            nutrition_data = {
                "Small": {"Calories": 180, "Protein": 8, "Fat": 6, "Carbs": 20},
                "Medium": {"Calories": 350, "Protein": 15, "Fat": 12, "Carbs": 45},
                "Large": {"Calories": 550, "Protein": 22, "Fat": 18, "Carbs": 65}
            }
            nutrients = nutrition_data[portion]

            # --- Health advice message ---
            if predicted_class == "Spoiled":
                health_msg = "🚫 Spoiled food — not safe to eat!"
            elif predicted_class == "Unhealthy":
                if nutrients["Calories"] > 450:
                    health_msg = "⚠️ High-calorie unhealthy meal — avoid large portions."
                else:
                    health_msg = "⚠️ Unhealthy but moderate calories — occasional treat."
            else:
                if nutrients["Calories"] < 250:
                    health_msg = "✅ Light & healthy meal."
                elif nutrients["Calories"] < 500:
                    health_msg = "✅ Balanced portion, healthy choice."
                else:
                    health_msg = "⚠️ Healthy food but large portion — eat moderately."

            # --- Final result dictionary ---
            result = {
                "prediction": predicted_class,
                "image_path": file_path,
                "portion": portion,
                "nutrients": nutrients,
                "health_msg": health_msg
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
