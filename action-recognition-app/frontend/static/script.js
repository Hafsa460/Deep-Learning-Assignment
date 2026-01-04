const API_BASE_URL = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predictBtn');
  if (predictBtn) predictBtn.addEventListener('click', onPredictClick);
});

async function onPredictClick() {
  const input = document.getElementById('imageInput');
  const errorDiv = document.getElementById('error');
  const resultDiv = document.getElementById('result');
  errorDiv.style.display = 'none';
  resultDiv.style.display = 'none';

  if (!input.files || input.files.length === 0) {
    errorDiv.textContent = 'Please select an image file.';
    errorDiv.style.display = 'block';
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch(API_BASE_URL + '/predict-single', { method: 'POST', body: formData });
    if (!resp.ok) {
      const j = await resp.json().catch(()=>({ error: 'Server error' }));
      throw new Error(j.error || 'Prediction failed');
    }
    const data = await resp.json();
    document.getElementById('predictedAction').textContent = (data.predicted_action || '-').replace(/_/g,' ');
    document.getElementById('confidence').textContent = data.confidence ? `Confidence: ${(data.confidence*100).toFixed(2)}%` : '';
    resultDiv.style.display = 'block';
  } catch (err) {
    errorDiv.textContent = err.message || 'Prediction error';
    errorDiv.style.display = 'block';
  }
}

