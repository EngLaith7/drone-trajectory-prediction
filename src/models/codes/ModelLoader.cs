// ModelLoader.cs
// Simple helper to load JSON model files from StreamingAssets using Newtonsoft.Json.
// Put rf_model.json inside Assets/StreamingAssets/

using System.IO;
using UnityEngine;
using Newtonsoft.Json;

public static class ModelLoader
{
    // relativePath example: "rf_model.json"
    public static T LoadFromStreamingAssets<T>(string relativePath)
    {
        string fullPath = Path.Combine(Application.streamingAssetsPath, relativePath);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"File not found: {fullPath}");

        string json = File.ReadAllText(fullPath);
        return JsonConvert.DeserializeObject<T>(json);
    }
}
