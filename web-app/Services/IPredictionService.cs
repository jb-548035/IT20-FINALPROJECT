// ============================================================
// Services/IPredictionService.cs
// Interface for the prediction service.
// Defining an interface allows easy unit-testing via mocks.
// ============================================================

using ShoppingPredictor.Models;

namespace ShoppingPredictor.Services
{
    public interface IPredictionService
    {
        /// <summary>
        /// Calls the FastAPI /predict endpoint and returns a structured result.
        /// </summary>
        Task<PredictionResult?> PredictAsync(CustomerInputModel input);

        /// <summary>
        /// Returns true if the FastAPI backend is reachable and healthy.
        /// </summary>
        Task<bool> IsApiHealthyAsync();
    }
}
