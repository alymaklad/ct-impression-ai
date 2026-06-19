import evaluationMetrics from "../../evaluation/evaluation_results.json" assert { type: "json" };
import modelManifest from "../../model/manifest.json" assert { type: "json" };

export const handler = async () => {
  return {
    statusCode: 200,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "public, max-age=300"
    },
    body: JSON.stringify({
      status: "ready",
      model: modelManifest,
      evaluation_metrics: evaluationMetrics
    })
  };
};

