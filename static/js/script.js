document.getElementById('goalForm').addEventListener('submit', function(e) {
    const target = document.getElementById('target_amount').value;
    const years = document.getElementById('years').value;
    if (target <= 0 || years <= 0) {
        alert('Please enter positive values for amount and years.');
        e.preventDefault();
    }
});