
from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow import keras
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)
# Load trained model
model = keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.get("/")
def home():
    return {"message": "MNIST CNN FastAPI is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)

    prediction = model.predict(img)
    predicted_class = int(np.argmax(prediction))

    return {
        "predicted_digit": predicted_class,
        "confidence": float(np.max(prediction))
    }
