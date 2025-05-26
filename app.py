# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# import torch
# import torchvision.transforms as transforms
# from model import HybridCNNGAT, create_edge_index

# # FastAPI app
# app = FastAPI(title="Thyroid Ultrasound Classification API")

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model = HybridCNNGAT().to(device)
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.eval()

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Health check
# @app.get("/")
# def index():
#     return {"message": "API is running. Use POST /predict/ to classify an image."}

# # Prediction endpoint
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         if not file.filename.endswith((".jpg", ".jpeg", ".png")):
#             return JSONResponse(content={"error": "Invalid image format. Please upload a .jpg or .png file."}, status_code=400)

#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         input_tensor = transform(image).unsqueeze(0).to(device)

#         edge_index = create_edge_index(1).to(device)  # Fixed: single-node self-loop

#         with torch.no_grad():
#             output = model(input_tensor, edge_index)
#             predicted_class = torch.argmax(output, dim=1).item()

#         classes = {0: "Benign", 1: "Malignant", 2: "Normal Thyroid"}
#         prediction = classes.get(predicted_class, "Unknown")

#         return {"prediction": prediction}

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import HybridCNNGAT, create_edge_index

app = FastAPI()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = HybridCNNGAT().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Health check
@app.get("/")
def read_root():
    return {"message": "Thyroid Ultrasound Classification API is running."}

# Predict endpoint for Flutter integration
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return JSONResponse(content={"error": "Invalid image format."}, status_code=400)

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # Edge index for a single node
        edge_index = create_edge_index(1).to(device)

        with torch.no_grad():
            output = model(tensor, edge_index)
            prediction_id = torch.argmax(output, dim=1).item()

        class_map = {0: "Benign", 1: "Malignant", 2: "Normal Thyroid"}
        prediction = class_map.get(prediction_id, "Unknown")

        return {"prediction": prediction, "class_id": prediction_id}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


