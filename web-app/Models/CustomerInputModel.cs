// ============================================================
// Models/CustomerInputModel.cs
// Represents the raw form data submitted by the user.
// DataAnnotations drive both server-side validation and
// Bootstrap form labels / error messages in the Razor view.
// ============================================================

using System.ComponentModel.DataAnnotations;

namespace ShoppingPredictor.Models
{
    public class CustomerInputModel
    {
        // ── Demographic ───────────────────────────────────────────────────
        [Required]
        [Range(10, 100)]
        [Display(Name = "Age")]
        public double Age { get; set; } = 28;

        [Required]
        [Range(0, 1_000_000)]
        [Display(Name = "Monthly Income (INR)")]
        public double MonthlyIncome { get; set; } = 45000;

        [Required]
        [Display(Name = "Gender")]
        public string Gender { get; set; } = "Male";

        [Required]
        [Display(Name = "City Tier")]
        public string CityTier { get; set; } = "Tier 1";

        // ── Digital Behavior ──────────────────────────────────────────────
        [Required]
        [Range(0, 24)]
        [Display(Name = "Daily Internet Hours")]
        public double DailyInternetHours { get; set; } = 5.5;

        [Required]
        [Range(0, 20)]
        [Display(Name = "Smartphone Usage (Years)")]
        public double SmartphoneUsageYears { get; set; } = 7;

        [Required]
        [Range(0, 24)]
        [Display(Name = "Social Media Hours / Day")]
        public double SocialMediaHours { get; set; } = 2.5;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Online Payment Trust Score (1–10)")]
        public double OnlinePaymentTrustScore { get; set; } = 8;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Tech Savvy Score (1–10)")]
        public double TechSavvyScore { get; set; } = 7;

        [Required]
        [Range(0, 100)]
        [Display(Name = "Monthly Online Orders")]
        public double MonthlyOnlineOrders { get; set; } = 12;

        // ── Shopping Behavior ─────────────────────────────────────────────
        [Required]
        [Range(0, 50)]
        [Display(Name = "Monthly Store Visits")]
        public double MonthlyStoreVisits { get; set; } = 2;

        [Required]
        [Range(0, 100_000)]
        [Display(Name = "Avg Online Spend (INR)")]
        public double AvgOnlineSpend { get; set; } = 3200;

        [Required]
        [Range(0, 100_000)]
        [Display(Name = "Avg Store Spend (INR)")]
        public double AvgStoreSpend { get; set; } = 800;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Discount Sensitivity (1–10)")]
        public double DiscountSensitivity { get; set; } = 7;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Return Frequency (1–10)")]
        public double ReturnFrequency { get; set; } = 4;

        [Required]
        [Range(0, 30)]
        [Display(Name = "Avg Delivery Days")]
        public double AvgDeliveryDays { get; set; } = 3;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Delivery Fee Sensitivity (1–10)")]
        public double DeliveryFeeSensitivity { get; set; } = 6;

        // ── Attitudinal ───────────────────────────────────────────────────
        [Required]
        [Range(1, 10)]
        [Display(Name = "Free Return Importance (1–10)")]
        public double FreeReturnImportance { get; set; } = 8;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Product Availability Online (1–10)")]
        public double ProductAvailabilityOnline { get; set; } = 8;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Impulse Buying Score (1–10)")]
        public double ImpulseBuyingScore { get; set; } = 6;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Need Touch / Feel Score (1–10)")]
        public double NeedTouchFeelScore { get; set; } = 3;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Brand Loyalty Score (1–10)")]
        public double BrandLoyaltyScore { get; set; } = 5;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Environmental Awareness (1–10)")]
        public double EnvironmentalAwareness { get; set; } = 7;

        [Required]
        [Range(1, 10)]
        [Display(Name = "Time Pressure Level (1–10)")]
        public double TimePressureLevel { get; set; } = 8;
    }
}
