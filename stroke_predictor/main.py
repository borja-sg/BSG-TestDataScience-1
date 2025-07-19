from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from stroke_predictor.utils.preprocessing import preprocess_dataframe
from stroke_predictor.utils.data_io import load_model_and_config
from stroke_predictor.utils.models import PatientFeatures
from stroke_predictor.utils.predictor import make_prediction


# Load the MLflow model and configurations
MODEL_PATH = "./models/stroke_predictor_optuna"
model, _, column_names, variable_encoder = load_model_and_config(MODEL_PATH)

templates = Jinja2Templates(directory="templates")

# Initialize FastAPI app
app = FastAPI(title="Stroke Predictor API")


@app.get("/")  # noqa
def read_root() -> dict:
    """Root endpoint to check if the API is running."""
    return {"message": "Stroke prediction model is up and running."}


@app.post("/predict")  # noqa
def predict(features: PatientFeatures) -> dict:
    """
    Predict stroke risk based on patient features.

    Parameters
    ----------
    features : BaseModel
        The patient features for stroke prediction.

    Returns
    -------
    dict
        A dictionary containing the stroke prediction and its probability.
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Preprocess the input data
    input_df = preprocess_dataframe(
        input_df, columns=column_names, variable_encoder=variable_encoder
    )

    print(f"Processed input data:\n{input_df}")

    # Run prediction using helper function
    prediction, probability = make_prediction(input_df, model)

    return {
        "stroke_prediction": prediction,
        "stroke_probability": round(probability, 4),
    }


@app.get("/web", response_class=HTMLResponse)  # noqa
async def get_form(request: Request) -> HTMLResponse:
    """
    Render the input form for stroke prediction.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    HTMLResponse
        The HTML response containing the input form."""
    return templates.TemplateResponse("form.html", {"request": request, "result": None})


@app.post("/web", response_class=HTMLResponse)  # noqa
async def predict_form(
    request: Request,
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    gender: str = Form(...),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    Residence_type: str = Form(...),
    smoking_status: str = Form(...),
) -> HTMLResponse:
    """
    Handle form submission for stroke prediction.

    Parameters
    ----------
    request : Request
        The FastAPI request object.
    age : float
        Age of the patient.
    hypertension : int
        Hypertension status (0 or 1).
    heart_disease : int
        Heart disease status (0 or 1).
    avg_glucose_level : float
        Average glucose level of the patient.
    bmi : float
        Body Mass Index of the patient.
    gender : Literal["Male", "Female", "Other"]:
        Gender of the patient.
    ever_married : Literal["Yes", "No"]
        Marital status of the patient.
    work_type : Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
        Type of work the patient is engaged in.
    Residence_type : Literal["Urban", "Rural"]
        Type of residence (Urban or Rural).
    smoking_status : Literal["formerly smoked", "never smoked", "smokes", "Unknown"]
        Smoking status of the patient.

    Returns
    -------
    HTMLResponse
        The HTML response containing the prediction result and input values.
    """
    # Build input DataFrame
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "gender": gender,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": Residence_type,
                "smoking_status": smoking_status,
            }
        ]
    )

    # Preprocess
    processed_input = preprocess_dataframe(
        input_data, columns=column_names, variable_encoder=variable_encoder
    )
    print(f"Processed input data:\n{processed_input}")
    prediction, probability = make_prediction(processed_input, model)

    if prediction == 1:
        result = f"The patient is likely to suffer a stroke with a probability of {probability:.2%}."
    else:
        result = f"It is unlikely that the patient will suffer a stroke with a probability of {probability:.2%}."

    input_values = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "smoking_status": smoking_status,
    }

    return templates.TemplateResponse(
        "form.html",
        {"request": request, "result": result, "input_values": input_values},
    )
