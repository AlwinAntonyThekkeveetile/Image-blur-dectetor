import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Mount static files (for CSS and other assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory for storing templates
templates = Jinja2Templates(directory="templates")

# Function to check if an image is blurry
def check_blurriness(image: np.ndarray) -> tuple[str, float]:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    if laplacian_var < 5:
        return "Image is blurry", laplacian_var
    else:
        return "Image is not blurry", laplacian_var

# Route to serve the upload form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to upload the image and check its blurriness
@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "The uploaded file is not a valid image."}

    blur_status, laplacian_var = check_blurriness(img)

    # Prepare the result message for the result.html template
    result_message = "Your image is blurry" if blur_status == "Image is blurry" else "Your image is not blurry"

    return templates.TemplateResponse("result.html", {
        "request": request, 
        "result_message": result_message
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
