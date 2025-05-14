from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import uvicorn


def create_app():
    app = FastAPI(
        title="CSV to API Converter",
        description="Upload CSV data and expose it via a RESTful API with filtering and search support.",
        version="1.0.0"
    )

    # In-memory storage for DataFrame
    app.state.df = None

    @app.post("/upload")
    async def upload_csv(file: UploadFile = File(...)):
        """
        Upload a CSV file. Replaces any previously uploaded data.
        """
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        try:
            contents = await file.read()
            df = pd.read_csv(pd.io.common.BytesIO(contents))
            app.state.df = df
            return {"message": f"CSV file '{file.filename}' uploaded successfully.", "rows": len(df)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN and infinite values with None for JSON serialization.
        """
        return df.replace({np.nan: None, np.inf: None, -np.inf: None})

    @app.get("/records")
    def get_records(request: Request, skip: int = 0, limit: int = 100):
        """
        Retrieve records with optional filtering. Use query parameters matching column names for exact filters.
        Example: /records?country=USA&skip=10&limit=5
        """
        df = app.state.df
        if df is None:
            raise HTTPException(status_code=404, detail="No CSV data uploaded yet.")

        # Start with full DataFrame
        results = df

        # Apply filters for each query param matching column names
        params = dict(request.query_params)
        params.pop('skip', None)
        params.pop('limit', None)
        for key, value in params.items():
            if key in df.columns:
                results = results[results[key] == _cast_type(df[key].dtype, value)]

        # Pagination
        paged = results.iloc[skip:skip + limit]

        # Clean dataframe for JSON
        paged_clean = _clean_df(paged)

        return JSONResponse(content={
            "total": len(results),
            "skip": skip,
            "limit": limit,
            "data": paged_clean.to_dict(orient='records')
        })

    @app.get("/search")
    def search(q: str, skip: int = 0, limit: int = 100):
        """
        Full-text search across all string columns.
        Example: /search?q=Apple&skip=0&limit=50
        """
        df = app.state.df
        if df is None:
            raise HTTPException(status_code=404, detail="No CSV data uploaded yet.")

        # Identify string columns
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        if not len(str_cols):
            return JSONResponse(content={"total": 0, "data": []})

        # Filter rows where any string column contains the query
        mask = False
        for col in str_cols:
            mask |= df[col].astype(str).str.contains(q, case=False, na=False)
        results = df[mask]

        # Pagination
        paged = results.iloc[skip:skip + limit]

        # Clean dataframe for JSON
        paged_clean = _clean_df(paged)

        return JSONResponse(content={
            "total": len(results),
            "skip": skip,
            "limit": limit,
            "data": paged_clean.to_dict(orient='records')
        })

    def _cast_type(dtype, value: str):
        """
        Helper to cast query parameter strings to the appropriate DataFrame column dtype.
        """
        try:
            if pd.api.types.is_integer_dtype(dtype):
                return int(value)
            if pd.api.types.is_float_dtype(dtype):
                return float(value)
        except ValueError:
            pass
        return value

    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
