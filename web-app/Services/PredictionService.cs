// ============================================================
// Services/PredictionService.cs
// Concrete implementation of IPredictionService.
// Sends customer features to the FastAPI /predict endpoint
// and deserialises the JSON response into PredictionResult.
// ============================================================

using System.Net.Http.Json;
using System.Text.Json;
using ShoppingPredictor.Models;

namespace ShoppingPredictor.Services
{
    public class PredictionService : IPredictionService
    {
        private readonly HttpClient _http;
        private readonly ILogger<PredictionService> _logger;

        // HttpClient is injected by the DI container (configured in Program.cs)
        public PredictionService(HttpClient http, ILogger<PredictionService> logger)
        {
            _http   = http;
            _logger = logger;
        }

        /// <inheritdoc/>
        public async Task<PredictionResult?> PredictAsync(CustomerInputModel input)
        {
            // ── Build the JSON payload that matches FastAPI's CustomerFeatures schema ──
            var payload = new
            {
                Age                         = input.Age,
                monthly_income              = input.MonthlyIncome,
                gender                      = input.Gender,
                city_tier                   = input.CityTier,
                daily_internet_hours        = input.DailyInternetHours,
                smartphone_usage_years      = input.SmartphoneUsageYears,
                social_media_hours          = input.SocialMediaHours,
                online_payment_trust_score  = input.OnlinePaymentTrustScore,
                tech_savvy_score            = input.TechSavvyScore,
                monthly_online_orders       = input.MonthlyOnlineOrders,
                monthly_store_visits        = input.MonthlyStoreVisits,
                avg_online_spend            = input.AvgOnlineSpend,
                avg_store_spend             = input.AvgStoreSpend,
                discount_sensitivity        = input.DiscountSensitivity,
                return_frequency            = input.ReturnFrequency,
                avg_delivery_days           = input.AvgDeliveryDays,
                delivery_fee_sensitivity    = input.DeliveryFeeSensitivity,
                free_return_importance      = input.FreeReturnImportance,
                product_availability_online = input.ProductAvailabilityOnline,
                impulse_buying_score        = input.ImpulseBuyingScore,
                need_touch_feel_score       = input.NeedTouchFeelScore,
                brand_loyalty_score         = input.BrandLoyaltyScore,
                environmental_awareness     = input.EnvironmentalAwareness,
                time_pressure_level         = input.TimePressureLevel
            };

            try
            {
                // POST to FastAPI /predict
                var response = await _http.PostAsJsonAsync("predict", payload);
                response.EnsureSuccessStatusCode();

                // Deserialise the JSON response into PredictionResult
                var result = await response.Content.ReadFromJsonAsync<PredictionResult>(
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                _logger.LogInformation(
                    "Prediction completed: {Class} ({Confidence:P1})",
                    result?.PredictedClass, result?.Confidence);

                return result;
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex,
                    "Failed to reach FastAPI prediction service at {BaseUrl}", _http.BaseAddress);
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error during prediction");
                return null;
            }
        }

        /// <inheritdoc/>
        public async Task<bool> IsApiHealthyAsync()
        {
            try
            {
                var response = await _http.GetAsync("health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
    }
}
