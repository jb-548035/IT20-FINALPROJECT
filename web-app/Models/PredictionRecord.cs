// ============================================================
// Models/PredictionRecord.cs
// Entity Framework Core entity.
// Each row = one saved prediction in the SQLite database.
// ============================================================

using System.ComponentModel.DataAnnotations;

namespace ShoppingPredictor.Models
{
    public class PredictionRecord
    {
        [Key]
        public int Id { get; set; }

        /// <summary>UTC timestamp of when the prediction was made.</summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        /// <summary>Predicted class: Online | Store | Hybrid.</summary>
        [Required, MaxLength(20)]
        public string PredictedClass { get; set; } = string.Empty;

        /// <summary>Model confidence (0–1).</summary>
        public double Confidence { get; set; }

        /// <summary>JSON-serialised probability dictionary.</summary>
        public string ProbabilitiesJson { get; set; } = "{}";

        /// <summary>Name of the ML model used.</summary>
        [MaxLength(100)]
        public string ModelUsed { get; set; } = string.Empty;

        // ── Raw input snapshot (for audit / debugging) ─────────────────────
        public double Age { get; set; }
        public double MonthlyIncome { get; set; }
        public string Gender { get; set; } = string.Empty;
        public string CityTier { get; set; } = string.Empty;
        public double MonthlyOnlineOrders { get; set; }
        public double AvgOnlineSpend { get; set; }
        public double AvgStoreSpend { get; set; }
        public double TechSavvyScore { get; set; }
        public double NeedTouchFeelScore { get; set; }
        public double DailyInternetHours { get; set; }
    }
}
