// ============================================================
// Models/ErrorViewModel.cs
// Used by the default /Home/Error action and view.
// ============================================================

namespace ShoppingPredictor.Models
{
    public class ErrorViewModel
    {
        public string? RequestId { get; set; }
        public bool ShowRequestId => !string.IsNullOrEmpty(RequestId);
        public string? Message { get; set; }
    }
}
