// ModelTest.cs
// Read a test CSV (StreamingAssets) where each row contains (features..., targets...).
// It computes MSE per target and overall RMSE and logs the result.
// Useful to verify RFModelRuntime predictions inside Unity.

using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class ModelTest : MonoBehaviour
{
    public string modelJsonFileName = "rf_model.json";
    public string testCsvFileName = "test_samples.csv"; // place in StreamingAssets
    public int featuresCount = 50; // window_size * featuresPerReading (adjust)
    public int targetsCount = 6; // posx,posy,posz,roll,pitch,yaw

    void Start()
    {
        try
        {
            RFModelRuntime rf = new RFModelRuntime(modelJsonFileName);
            RunTest(rf);
        }
        catch (System.Exception ex)
        {
            Debug.LogError("ModelTest: model load failed: " + ex.Message);
        }
    }

    void RunTest(RFModelRuntime rf)
    {
        string path = Path.Combine(Application.streamingAssetsPath, testCsvFileName);
        if (!File.Exists(path)) { Debug.LogError("Test CSV not found: " + path); return; }

        string[] lines = File.ReadAllLines(path);
        List<double[]> yTrue = new List<double[]>();
        List<double[]> yPred = new List<double[]>();

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] toks = line.Split(',');
            if (toks.Length < featuresCount + targetsCount) continue;

            float[] x = new float[featuresCount];
            for (int i = 0; i < featuresCount; i++) x[i] = float.Parse(toks[i].Trim());

            float[] pred = rf.Predict(x);
            double[] predD = new double[targetsCount];
            double[] trueD = new double[targetsCount];

            for (int k = 0; k < targetsCount; k++)
            {
                predD[k] = pred[k];
                trueD[k] = double.Parse(toks[featuresCount + k].Trim());
            }
            yPred.Add(predD);
            yTrue.Add(trueD);
        }

        if (yPred.Count == 0) { Debug.LogWarning("No test rows found."); return; }

        int N = yPred.Count;
        int M = targetsCount;
        double[] seSum = new double[M];
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < M; k++)
            {
                double diff = yTrue[i][k] - yPred[i][k];
                seSum[k] += diff * diff;
            }
        }

        // compute MSE and RMSE per output
        string report = $"ModelTest: samples={N}\n";
        double totalMse = 0;
        for (int k = 0; k < M; k++)
        {
            double mse = seSum[k] / N;
            double rmse = System.Math.Sqrt(mse);
            report += $" output[{k}] MSE={mse:F6} RMSE={rmse:F6}\n";
            totalMse += mse;
        }
        report += $" avg MSE={totalMse / M:F6}";
        Debug.Log(report);
    }
}
