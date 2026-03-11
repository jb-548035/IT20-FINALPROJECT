# Migrations

This folder contains Entity Framework Core migration files
for the SQLite database.

## How to generate migrations

From the `web-app/` directory, run:

```bash
# Install EF CLI (once)
dotnet tool install --global dotnet-ef

# Add first migration (creates PredictionRecords table)
dotnet ef migrations add InitialCreate

# Apply migrations to the database
dotnet ef database update
```

> **Note:** The application also calls `db.Database.EnsureCreated()` at
> startup (in `Program.cs`), so for development you don't *need* to run
> migrations manually — the SQLite file is created automatically.
> Use the EF migration approach for production deployments.
