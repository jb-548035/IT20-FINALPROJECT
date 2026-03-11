// ============================================================
// Models/PredictionResult.cs
// Represents the JSON response returned by the FastAPI backend.
// System.Text.Json deserializes the API response into this class.
// ============================================================

using System.Text.Json.Serialization;

namespace ShoppingPredictor.Models
{
    /// <summary>
    /// Maps the JSON body returned by POST /predict on the FastAPI service.
    /// </summary>
    public class PredictionResult
    {
        /// <summary>Predicted shopping preference: "Online", "Store", or "Hybrid".</summary>
        [JsonPropertyName("predicted_class")]
        public string PredictedClass { get; set; } = string.Empty;

        /// <summary>Probability (0–1) of the winning class.</summary>
        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        /// <summary>Full probability distribution across all three classes.</summary>
        [JsonPropertyName("probabilities")]
        public Dictionary<string, double> Probabilities { get; set; } = new();

        /// <summary>Name of the model that produced the prediction (e.g., "LogisticRegression").</summary>
        [JsonPropertyName("model_used")]
        public string ModelUsed { get; set; } = string.Empty;

        // ── Derived / computed properties (not from JSON) ─────────────────

        /// <summary>Confidence expressed as a percentage string, e.g. "92.1%".</summary>
        public string ConfidencePercent => $"{Confidence * 100:F1}%";

        /// <summary>Bootstrap badge colour class for the predicted class.</summary>
        public string BadgeClass => PredictedClass switch
        {
            "Online" => "bg-primary",
            "Store"  => "bg-warning text-dark",
            "Hybrid" => "bg-success",
            _        => "bg-secondary"
        };

        /// <summary>Emoji icon for the predicted class.</summary>
        public string Icon => PredictedClass switch
        {
            "Online" => "💻",
            "Store"  => "🏪",
            "Hybrid" => "🔄",
            _        => "❓"
        };

        /// <summary>Business recommendation copy for each class.</summary>
        public string BusinessInsight => PredictedClass switch
        {
            "Online" =>
                "This customer strongly prefers digital channels. " +
                "Invest in personalised email campaigns, app push notifications, " +
                "fast delivery options, and a frictionless mobile checkout experience.",
            "Store" =>
                "This customer prefers the physical retail experience. " +
                "Focus on in-store promotions, loyalty cards, visual merchandising, " +
                "and personalised assistance from store staff.",
            "Hybrid" =>
                "This customer actively uses both channels. " +
                "Deploy an omni-channel strategy: click-and-collect, unified loyalty points, " +
                "and cross-channel personalised recommendations to maximise engagement.",
            _ => "Insufficient data to generate a business recommendation."
        };
    }
}
