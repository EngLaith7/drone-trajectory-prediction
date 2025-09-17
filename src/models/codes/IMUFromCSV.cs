// IMUFromCSV.cs
// Streams IMU readings from a CSV file (StreamingAssets).
// Supports CSV with or without a leading `time` column.
// Produces a sliding flattened window identical to SimulatedIMU:
// flattened length = windowSize * featuresPerReading
// Default: featuresPerReading = 9 (accel3, gyro3, mag3)

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Globalization;

[RequireComponent(typeof(Transform))]
public class IMUFromCSV : MonoBehaviour
{
    [Header("CSV and playback")]
    public string csvFileName = "imu_data.csv"; // put file in StreamingAssets
    public int windowSize = 5;
    public int featuresPerReading = 9; // model expects 9 features per reading
    [Tooltip("If true, skip the first CSV column (commonly a time column) and use the next columns as features.")]
    public bool skipFirstColumnForTime = true; // default true
    public float sampleInterval = 0.02f; // playback rate

    private List<float[]> allRows = new List<float[]>();
    private int playIndex = 0;
    private float[,] buffer;
    private int bufferIndex = 0;
    private int filled = 0;
    private float[] lastFlatWindow = null;
    public float[] GetLatestFlatWindow() => lastFlatWindow;

    public delegate void OnWindowReady(float[] flat);
    public event OnWindowReady OnWindowReadyEvent;

    void Start()
    {
        LoadCsv();
        buffer = new float[windowSize, featuresPerReading];
        StartCoroutine(PlayLoop());
    }

    bool TryParseFloat(string s, out float val)
    {
        return float.TryParse(s.Trim(), NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out val);
    }

    void LoadCsv()
    {
        string path = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (!File.Exists(path))
        {
            Debug.LogError($"CSV not found: {path}");
            return;
        }
        string[] lines = File.ReadAllLines(path);
        if (lines.Length == 0) { Debug.LogWarning("CSV is empty."); return; }

        int lineNum = 0;
        foreach (var rawLine in lines)
        {
            lineNum++;
            if (string.IsNullOrWhiteSpace(rawLine)) continue;

            string[] toks = rawLine.Split(',');

            int expectedTokens = skipFirstColumnForTime ? featuresPerReading + 1 : featuresPerReading;
            if (toks.Length < expectedTokens)
            {
                Debug.LogWarning($"Skipping line {lineNum}: not enough columns ({toks.Length}) for expected {expectedTokens}.");
                continue;
            }

            // Detect header-like rows (non-numeric where numeric expected)
            bool looksLikeHeader = false;
            int checkStart = skipFirstColumnForTime ? 1 : 0;
            for (int i = checkStart; i < checkStart + Mathf.Min(3, featuresPerReading); i++)
            {
                float tmp;
                if (!TryParseFloat(toks[i], out tmp))
                {
                    looksLikeHeader = true;
                    break;
                }
            }
            if (looksLikeHeader) continue;

            float[] vals = new float[featuresPerReading];
            int srcStart = skipFirstColumnForTime ? 1 : 0;
            bool rowOk = true;
            for (int i = 0; i < featuresPerReading; i++)
            {
                float v;
                if (!TryParseFloat(toks[srcStart + i], out v))
                {
                    rowOk = false;
                    break;
                }
                vals[i] = v;
            }
            if (!rowOk)
            {
                Debug.LogWarning($"Skipping line {lineNum}: parse error.");
                continue;
            }
            allRows.Add(vals);
        }

        Debug.Log($"[IMUFromCSV] Loaded {allRows.Count} rows from CSV (featuresPerReading={featuresPerReading}, skipTimeColumn={skipFirstColumnForTime}).");
    }

    IEnumerator PlayLoop()
    {
        if (allRows.Count == 0) yield break;
        while (true)
        {
            float[] row = allRows[playIndex];
            int i = bufferIndex;
            for (int f = 0; f < featuresPerReading; f++) buffer[i, f] = row[f];
            bufferIndex = (bufferIndex + 1) % windowSize;
            if (filled < windowSize) filled++;
            if (filled == windowSize)
            {
                float[] flat = new float[windowSize * featuresPerReading];
                int idx = 0;
                int start = bufferIndex;
                for (int w = 0; w < windowSize; w++)
                {
                    int r = (start + w) % windowSize;
                    for (int f = 0; f < featuresPerReading; f++)
                        flat[idx++] = buffer[r, f];
                }
                lastFlatWindow = (float[])flat.Clone();
                OnWindowReadyEvent?.Invoke(flat);
            }

            playIndex = (playIndex + 1) % allRows.Count;
            yield return new WaitForSeconds(sampleInterval);
        }
    }
}
