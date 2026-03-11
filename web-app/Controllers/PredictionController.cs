// ============================================================
// Controllers/PredictionController.cs
// Manages the end-to-end prediction workflow:
//   GET  /Prediction/Predict  → shows the input form
//   POST /Prediction/Predict  → submits features, calls FastAPI, shows result
//   GET  /Prediction/History  → lists all saved predictions
//   POST /Prediction/Clear    → deletes all prediction records
// ============================================================

using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Text.Json;
using ShoppingPredictor.Data;
using ShoppingPredictor.Models;
using ShoppingPredictor.Services;

namespace ShoppingPredictor.Controllers
{
    public class PredictionController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly AppDbContext       _db;
        private readonly ILogger<PredictionController> _logger;

        public PredictionController(
            IPredictionService predictionService,
            AppDbContext db,
            ILogger<PredictionController> logger)
        {
            _predictionService = predictionService;
            _db                = db;
            _logger            = logger;
        }

        // ── GET /Prediction/Predict ────────────────────────────────────────
        // Shows the empty prediction form with default values
        [HttpGet]
        public IActionResult Predict()
        {
            return View(new CustomerInputModel());
        }

        // ── POST /Prediction/Predict ───────────────────────────────────────
        // 1. Validates form data (via DataAnnotations on CustomerInputModel)
        // 2. Calls FastAPI through PredictionService
        // 3. Saves record to SQLite
        // 4. Passes result to the Result partial/view
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Predict(CustomerInputModel input)
        {
            // Server-side validation
            if (!ModelState.IsValid)
            {
                return View(input);
            }

            // Call the ML API
            var result = await _predictionService.PredictAsync(input);

            if (result is null)
            {
                ModelState.AddModelError(string.Empty,
                    "⚠️ Could not reach the prediction API. " +
                    "Make sure the FastAPI service (py-app) is running on port 8000.");
                return View(input);
            }

            // ── Persist prediction to SQLite ──────────────────────────────
            var record = new PredictionRecord
            {
                Timestamp           = DateTime.UtcNow,
                PredictedClass      = result.PredictedClass,
                Confidence          = result.Confidence,
                ProbabilitiesJson   = JsonSerializer.Serialize(result.Probabilities),
                ModelUsed           = result.ModelUsed,
                Age                 = input.Age,
                MonthlyIncome       = input.MonthlyIncome,
                Gender              = input.Gender,
                CityTier            = input.CityTier,
                MonthlyOnlineOrders = input.MonthlyOnlineOrders,
                AvgOnlineSpend      = input.AvgOnlineSpend,
                AvgStoreSpend       = input.AvgStoreSpend,
                TechSavvyScore      = input.TechSavvyScore,
                NeedTouchFeelScore  = input.NeedTouchFeelScore,
                DailyInternetHours  = input.DailyInternetHours
            };

            _db.PredictionRecords.Add(record);
            await _db.SaveChangesAsync();

            _logger.LogInformation(
                "Saved prediction #{Id}: {Class} ({Conf:P1})",
                record.Id, record.PredictedClass, record.Confidence);

            // Pass both the input and result to the view
            ViewBag.Input = input;
            return View("Result", result);
        }

        // ── GET /Prediction/History ────────────────────────────────────────
        // Shows a paginated table of all past predictions
        [HttpGet]
        public async Task<IActionResult> History(int page = 1)
        {
            const int pageSize = 20;

            var total   = await _db.PredictionRecords.CountAsync();
            var records = await _db.PredictionRecords
                .OrderByDescending(r => r.Timestamp)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            // Summary statistics for the dashboard cards
            ViewBag.Total        = total;
            ViewBag.Page         = page;
            ViewBag.PageSize     = pageSize;
            ViewBag.TotalPages   = (int)Math.Ceiling(total / (double)pageSize);
            ViewBag.AvgConfidence = total > 0
                ? _db.PredictionRecords.Average(r => r.Confidence) * 100
                : 0.0;
            ViewBag.ClassCounts  = _db.PredictionRecords
                .GroupBy(r => r.PredictedClass)
                .Select(g => new { Class = g.Key, Count = g.Count() })
                .ToList();

            return View(records);
        }

        // ── POST /Prediction/Clear ─────────────────────────────────────────
        // Deletes all prediction records (reset button in History view)
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Clear()
        {
            _db.PredictionRecords.RemoveRange(_db.PredictionRecords);
            await _db.SaveChangesAsync();
            TempData["Success"] = "Prediction history cleared successfully.";
            return RedirectToAction(nameof(History));
        }
    }
}
