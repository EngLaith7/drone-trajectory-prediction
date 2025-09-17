// DataLogger.cs
// Subscribe to IMU OnWindowReady (SimulatedIMU or IMUFromCSV) and write flat window + current ground-truth pose to CSV.
// The CSV will be written to Application.persistentDataPath.

using System.IO;
using UnityEngine;

public class DataLogger : MonoBehaviour
{
    public string outFileName = "imu_logged.csv";
    public bool append = false;
    public bool writeHeader = true;

    private StreamWriter writer;
    private SimulatedIMU simImu;
    private IMUFromCSV csvImu;
    private Rigidbody rb;

    void Start()
    {
        simImu = GetComponent<SimulatedIMU>();
        csvImu = GetComponent<IMUFromCSV>();
        rb = GetComponent<Rigidbody>();

        string path = Path.Combine(Application.persistentDataPath, outFileName);
        writer = new StreamWriter(path, append);

        int windowSize = 5;
        int feat = 10;
        if (simImu != null) { windowSize = simImu.windowSize; feat = simImu.featuresPerReading; }
        else if (csvImu != null) { windowSize = csvImu.windowSize; feat = csvImu.featuresPerReading; }

        if (writeHeader)
        {
            string[] cols = new string[windowSize * feat + 6];
            for (int i = 0; i < cols.Length - 6; i++) cols[i] = $"f{i}";
            cols[cols.Length - 6] = "posx"; cols[cols.Length - 5] = "posy"; cols[cols.Length - 4] = "posz";
            cols[cols.Length - 3] = "roll"; cols[cols.Length - 2] = "pitch"; cols[cols.Length - 1] = "yaw";
            writer.WriteLine(string.Join(",", cols));
        }

        // subscribe
        if (simImu != null) simImu.OnWindowReadyEvent += OnWindowReady;
        if (csvImu != null) csvImu.OnWindowReadyEvent += OnWindowReady;
    }

    private void OnWindowReady(float[] flat)
    {
        Vector3 p = rb != null ? rb.position : Vector3.zero;
        Vector3 e = rb != null ? rb.rotation.eulerAngles : Vector3.zero;
        string[] tokens = new string[flat.Length + 6];
        for (int i = 0; i < flat.Length; i++) tokens[i] = flat[i].ToString("F6");
        tokens[flat.Length + 0] = p.x.ToString("F6");
        tokens[flat.Length + 1] = p.y.ToString("F6");
        tokens[flat.Length + 2] = p.z.ToString("F6");
        tokens[flat.Length + 3] = e.x.ToString("F6");
        tokens[flat.Length + 4] = e.y.ToString("F6");
        tokens[flat.Length + 5] = e.z.ToString("F6");
        writer.WriteLine(string.Join(",", tokens));
        writer.Flush();
    }

    void OnDestroy()
    {
        if (writer != null) writer.Close();
    }
}
