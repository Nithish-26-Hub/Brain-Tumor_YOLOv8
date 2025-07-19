$(document).ready(function () {
    let selectedFile = null;

    // File Input Change Event
    $("#fileInput").change(function (event) {
        selectedFile = event.target.files[0];

        if (selectedFile) {
            let reader = new FileReader();
            reader.onload = function (e) {
                $("#previewImage").attr("src", e.target.result).show();
                $("#predictBtn").prop("disabled", false);
            };
            reader.readAsDataURL(selectedFile);
        } else {
            $("#previewImage").hide();
            $("#predictBtn").prop("disabled", true);
        }
    });

    // Predict Button Click Event
    $("#predictBtn").click(function () {
        if (!selectedFile) {
            alert("Please upload an image first.");
            return;
        }

        let formData = new FormData();
        formData.append("file", selectedFile);

        // Send AJAX Request
        $.ajax({
            url: "/predict",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            beforeSend: function () {
                $("#predictBtn").text("⏳ Analyzing...").prop("disabled", true);
            },
            success: function (response) {
                $("#resultBox").show();
                $("#predictionText").text(`📢 ${response.prediction}`);
                $("#confidenceText").text(`🔬 Confidence: ${response.confidence}%`);
            },
            error: function () {
                alert("Error processing the image. Please try again.");
            },
            complete: function () {
                $("#predictBtn").text("🔍 Predict").prop("disabled", false);
            }
        });
    });
});
