// ============================================================
// Controllers/HomeController.cs
// Handles the landing page and static informational views.
// ============================================================

using Microsoft.AspNetCore.Mvc;
using ShoppingPredictor.Data;
using ShoppingPredictor.Models;
using ShoppingPredictor.Services;
using System.Diagnostics;

namespace ShoppingPredictor.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly AppDbContext _db;
        private readonly IPredictionService _predictionService;

        public HomeController(
            ILogger<HomeController> logger,
            AppDbContext db,
            IPredictionService predictionService)
        {
            _logger            = logger;
            _db                = db;
            _predictionService = predictionService;
        }

        // GET /
        // GET /Home/Index
        public async Task<IActionResult> Index()
        {
            // Check whether the FastAPI backend is reachable
            ViewBag.ApiHealthy      = await _predictionService.IsApiHealthyAsync();
            ViewBag.TotalPredictions = _db.PredictionRecords.Count();
            return View();
        }

        // GET /Home/About
        public IActionResult About() => View();

        // GET /Home/Error
        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel
            {
                RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier
            });
        }
    }
}
