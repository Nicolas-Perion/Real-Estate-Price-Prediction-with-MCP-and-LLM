from fastmcp import FastMCP
import joblib
import pandas as pd
from pipeline import PropertyPreprocessor
from typing import Optional

# Loading the pipeline
try:
    pipeline = joblib.load("price_prediction_model.joblib")
    print("Pipeline loaded")
except Exception as e:
    print(f"Error: {e}")
    pipeline = None

# Creating the MCP Server
mcp = FastMCP("Real Estate Predictor")

# Default values in case the user doesn't mention them (the prediction model still needs those arguments to make a prediction)
DEFAULTS = {
    "bedrooms": "2",
    "bathrooms": "1",
    "property_type": "Apartment"
}

# Tools (that the LLM will call and use)
@mcp.tool()
def estimate_price(area_value: float, city: str, bedrooms: Optional[str] = None, bathrooms: Optional[str] = None, property_type: Optional[str] = None) -> str:
    """
    Estimates the price of a property.
    """
    # Set default values if the question of the user didn't include values for those arguments
    bedrooms = bedrooms if bedrooms is not None else DEFAULTS["bedrooms"]
    bathrooms = bathrooms if bathrooms is not None else DEFAULTS["bathrooms"]
    property_type = property_type if property_type is not None else DEFAULTS["property_type"]
    
    if pipeline is None:
        return "Model not available."
    try:
        input_df = pd.DataFrame([{
            "area_value": min(area_value, 5000),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_type": property_type,
            "city": city.title()
        }])
        price = pipeline.predict(input_df)[0]
        return f"Estimated price: {price:,.0f} €"
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Strat the server
if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)