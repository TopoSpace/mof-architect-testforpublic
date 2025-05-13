document.addEventListener("DOMContentLoaded", function() {
    const predictButton = document.getElementById("predict_button");
    const smilesInput = document.getElementById("smiles_input");
    const resultText = document.getElementById("result_text");

    if (predictButton) {
        predictButton.addEventListener("click", function() {
            const smiles = smilesInput.value.trim();
            if (smiles === "") {
                resultText.textContent = "Please enter a SMILES string.";
                resultText.style.color = "red"; // Optional: style error messages
                return;
            }

            resultText.textContent = "Predicting...";
            resultText.style.color = ""; // Reset color

            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ "smiles": smiles }),
            })
            .then(response => {
                // Check if the response is ok (status in the range 200-299)
                if (response.ok) {
                    return response.json(); // Parse JSON if response is ok
                } else {
                    // If response is not ok, try to parse the error message from the JSON body
                    return response.json().then(errorData => {
                        // Construct an error object that includes the backend message
                        let error = new Error(errorData.error || `HTTP error! status: ${response.status}`);
                        error.response = response; // Attach the response for further inspection if needed
                        error.errorData = errorData; // Attach parsed error data
                        throw error; // Throw to be caught by the .catch block
                    }).catch(jsonParseError => {
                        // If parsing JSON fails (e.g. backend sent non-JSON error or network issue)
                        // Fallback to a generic error based on status
                        let error = new Error(`HTTP error! status: ${response.status}. Could not parse error response.`);
                        error.response = response;
                        throw error;
                    });
                }
            })
            .then(data => {
                // This block is now only for successful (response.ok) predictions
                // The backend structure for success is { "prediction": "some_value" }
                // The backend structure for handled errors (like invalid SMILES) is { "error": "message" },
                // which is now handled by the !response.ok path above.
                resultText.textContent = data.prediction;
                resultText.style.color = "green"; // Optional: style success messages
            })
            .catch(error => {
                console.error("Error during prediction:", error);
                // Display the specific error message from the backend if available, otherwise a generic one
                if (error.errorData && error.errorData.error) {
                    resultText.textContent = `Error: ${error.errorData.error}`;
                } else {
                    resultText.textContent = error.message || "An error occurred while fetching the prediction. Please check the console.";
                }
                resultText.style.color = "red"; // Optional: style error messages
            });
        });
    }
});

