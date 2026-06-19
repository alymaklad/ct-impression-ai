const formatMetric = (value) => {
  if (typeof value !== "number") return "N/A";
  return value.toFixed(3);
};

const setText = (id, value) => {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
};

try {
  const response = await fetch("/api/model-health");
  const data = await response.json();
  const metrics = data.evaluation_metrics ?? {};

  setText("rougeL", formatMetric(metrics["ROUGE-L"]));
  setText("bertScore", formatMetric(metrics["BERTScore-F1"]));
  setText("clinicalCoverage", formatMetric(metrics["Clinical-Keyword-Coverage"]));
} catch {
  setText("rougeL", "N/A");
  setText("bertScore", "N/A");
  setText("clinicalCoverage", "N/A");
}

