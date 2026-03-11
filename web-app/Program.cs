// ============================================================
// Program.cs
// Entry point for the ASP.NET Core MVC web application.
// Configures services (DI container) and the HTTP pipeline.
// ============================================================

using Microsoft.EntityFrameworkCore;
using ShoppingPredictor.Data;
using ShoppingPredictor.Services;

var builder = WebApplication.CreateBuilder(args);

// ── 1. Add MVC (Controllers + Razor Views) ────────────────────────────────
builder.Services.AddControllersWithViews();

// ── 2. Register SQLite via Entity Framework Core ─────────────────────────
//    The connection string is read from appsettings.json → "DefaultConnection"
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// ── 3. Register the ML prediction service (calls FastAPI) ────────────────
//    HttpClient is registered as a named/typed client for dependency injection
builder.Services.AddHttpClient<IPredictionService, PredictionService>(client =>
{
    // Base URL of the FastAPI backend (py-app); configurable via appsettings
    client.BaseAddress = new Uri(
        builder.Configuration["PredictionApi:BaseUrl"] ?? "http://localhost:8000/");
    client.Timeout = TimeSpan.FromSeconds(30);
});

// ── 4. Session support (used for temporary UI state) ─────────────────────
builder.Services.AddSession(options =>
{
    options.IdleTimeout = TimeSpan.FromMinutes(30);
    options.Cookie.HttpOnly = true;
    options.Cookie.IsEssential = true;
});

var app = builder.Build();

// ── 5. HTTP pipeline ──────────────────────────────────────────────────────
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();   // Serves wwwroot/ (CSS, JS, images)
app.UseRouting();
app.UseSession();
app.UseAuthorization();

// ── 6. Default route: HomeController → Index action ──────────────────────
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

// ── 7. Ensure database is created on startup ──────────────────────────────
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.EnsureCreated();   // Creates tables if they don't exist
}

app.Run();
