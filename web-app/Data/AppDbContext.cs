// ============================================================
// Data/AppDbContext.cs
// Entity Framework Core DbContext.
// Manages the SQLite database connection and table mappings.
// ============================================================

using Microsoft.EntityFrameworkCore;
using ShoppingPredictor.Models;

namespace ShoppingPredictor.Data
{
    public class AppDbContext : DbContext
    {
        // Constructor receives options (connection string) injected by DI
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        /// <summary>Table: PredictionRecords — stores every prediction made via the UI.</summary>
        public DbSet<PredictionRecord> PredictionRecords { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Index on Timestamp for efficient "recent predictions" queries
            modelBuilder.Entity<PredictionRecord>()
                .HasIndex(p => p.Timestamp)
                .HasDatabaseName("IX_PredictionRecords_Timestamp");

            // Index on PredictedClass for distribution queries
            modelBuilder.Entity<PredictionRecord>()
                .HasIndex(p => p.PredictedClass)
                .HasDatabaseName("IX_PredictionRecords_PredictedClass");
        }
    }
}
