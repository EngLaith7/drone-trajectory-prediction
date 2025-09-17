// RFModelRuntime.cs
// Runtime RandomForest predictor (reads the exported JSON structure).
// The JSON is expected to contain:
//  - n_estimators, n_outputs, forest (list of trees)
//  - each tree: n_nodes, children_left, children_right, feature, threshold, value
//  - scaler_X: mean, scale
//  - scaler_y: mean, scale
//
// The tree.value is expected as a list per node: value[node] -> [out0, out1, ..., outN]

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

[Serializable]
public class RFPayload
{
    public int n_estimators;
    public int n_outputs;
    public List<TreeNodeArray> forest;
    public ScalerPayload scaler_X;
    public ScalerPayload scaler_y;
}

[Serializable]
public class ScalerPayload
{
    public List<float> mean;
    public List<float> scale;
}

[Serializable]
public class TreeNodeArray
{
    public int n_nodes;
    public List<int> children_left;
    public List<int> children_right;
    public List<int> feature;
    public List<float> threshold;
    public List<List<float>> value; // value[node] -> list of outputs
}

public class RFModelRuntime
{
    private RFPayload payload;

    // Read and validate JSON at construction
    public RFModelRuntime(string jsonRelativePath)
    {
        string fullPath = Path.Combine(Application.streamingAssetsPath, jsonRelativePath);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"RF JSON not found: {fullPath}");

        string json = File.ReadAllText(fullPath);
        payload = JsonConvert.DeserializeObject<RFPayload>(json);

        if (payload == null)
            throw new Exception("Failed to deserialize RF payload.");

        if (payload.scaler_X == null || payload.scaler_y == null)
            throw new Exception("Missing scaler information in JSON.");

        if (payload.forest == null || payload.forest.Count == 0)
            throw new Exception("Empty forest in JSON.");

        // sanity checks for tree sizes
        foreach (var tree in payload.forest)
        {
            if (tree.children_left.Count != tree.n_nodes || tree.children_right.Count != tree.n_nodes)
                throw new Exception("Tree children arrays length mismatch with n_nodes.");
            if (tree.feature.Count != tree.n_nodes || tree.threshold.Count != tree.n_nodes || tree.value.Count != tree.n_nodes)
                throw new Exception("Tree node arrays length mismatch.");
        }

        Debug.Log($"[RFModelRuntime] Loaded RF: estimators={payload.n_estimators}, outputs={payload.n_outputs}, input_features={payload.scaler_X.mean.Count}");
    }

    // Standardize input vector using scaler_X (x - mean) / scale
    private float[] StandardizeX(float[] x)
    {
        int n = payload.scaler_X.mean.Count;
        if (x.Length != n)
            throw new ArgumentException($"Input length {x.Length} does not match model expected {n}.");

        float[] outp = new float[n];
        for (int i = 0; i < n; i++)
            outp[i] = (x[i] - payload.scaler_X.mean[i]) / payload.scaler_X.scale[i];
        return outp;
    }

    // Inverse transform predicted y (yscaled * scale + mean)
    private float[] InverseY(float[] yscaled)
    {
        int n = payload.scaler_y.mean.Count;
        float[] outp = new float[n];
        for (int i = 0; i < n; i++)
            outp[i] = yscaled[i] * payload.scaler_y.scale[i] + payload.scaler_y.mean[i];
        return outp;
    }

    // Predict: x_flat = flattened window (length must match scaler_X.mean.Count)
    // returns output array length payload.n_outputs
    public float[] Predict(float[] x_flat)
    {
        float[] xstd = StandardizeX(x_flat);
        int outputs = payload.n_outputs;
        double[] sumOutputs = new double[outputs];

        // For each tree, traverse to leaf and accumulate leaf values
        foreach (var tree in payload.forest)
        {
            int node = 0;
            while (true)
            {
                int left = tree.children_left[node];
                int right = tree.children_right[node];

                // leaf if both children == -1 (typical sklearn)
                if (left == -1 && right == -1)
                {
                    var vals = tree.value[node]; // list of outputs
                    if (vals.Count != outputs)
                        throw new Exception("Tree node value length does not match n_outputs.");
                    for (int k = 0; k < outputs; k++)
                        sumOutputs[k] += vals[k];
                    break;
                }
                else
                {
                    int feat = tree.feature[node];
                    float thr = tree.threshold[node];
                    if (xstd[feat] <= thr) node = left;
                    else node = right;
                }
            }
        }

        // average across estimators
        float[] avg = new float[outputs];
        for (int k = 0; k < outputs; k++) avg[k] = (float)(sumOutputs[k] / payload.n_estimators);

        // inverse transform to original units
        float[] y = InverseY(avg);
        return y;
    }
}
