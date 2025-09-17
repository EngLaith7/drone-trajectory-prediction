// SimulatedIMU.cs
// Produces IMU readings from the Rigidbody physics (accel, gyro, mag).
// Builds a circular window buffer and exposes the latest flattened window.

using System.Collections;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class SimulatedIMU : MonoBehaviour
{
    [Header("IMU Settings")]
    public int windowSize = 5;
    public int featuresPerReading = 9; // must match model training (9)
    public float sampleInterval = 0.02f; // seconds, default 50 Hz
    public Vector3 worldMagneticField = new Vector3(0.2f, 0f, 0.5f);
    public bool gyroInDegrees = true; // convert rad/s -> deg/s if true

    Rigidbody rb;
    Vector3 prevVelocity;
    float[,] buffer;
    int bufferIndex = 0;
    int filled = 0;
    private float[] lastFlatWindow = null;
    public float[] GetLatestFlatWindow() { return lastFlatWindow; }

    public delegate void OnWindowReady(float[] flatWindow);
    public event OnWindowReady OnWindowReadyEvent;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        prevVelocity = rb.linearVelocity;
        buffer = new float[windowSize, featuresPerReading];
        StartCoroutine(SampleLoop());
    }

    IEnumerator SampleLoop()
    {
        while (true)
        {
            SampleOnce();
            yield return new WaitForSeconds(sampleInterval);
        }
    }

    void SampleOnce()
    {
        float dt = sampleInterval;
        Vector3 v = rb.linearVelocity;
        Vector3 worldAcc = (v - prevVelocity) / Mathf.Max(dt, 1e-6f);
        prevVelocity = v;

        // proper acceleration = acceleration - gravity
        Vector3 properAccWorld = worldAcc - Physics.gravity;
        Vector3 acc_body = transform.InverseTransformDirection(properAccWorld);

        // angular velocity (rad/s) from Rigidbody
        Vector3 angVelWorld = rb.angularVelocity;
        Vector3 gyro_body = transform.InverseTransformDirection(angVelWorld);
        if (gyroInDegrees) gyro_body *= Mathf.Rad2Deg;

        Vector3 mag_body = transform.InverseTransformDirection(worldMagneticField.normalized);

        int i = bufferIndex;
        // fill buffer with 9 features: accel(3), gyro(3), mag(3)
        buffer[i, 0] = acc_body.x;
        buffer[i, 1] = acc_body.y;
        buffer[i, 2] = acc_body.z;
        buffer[i, 3] = gyro_body.x;
        buffer[i, 4] = gyro_body.y;
        buffer[i, 5] = gyro_body.z;
        buffer[i, 6] = mag_body.x;
        buffer[i, 7] = mag_body.y;
        buffer[i, 8] = mag_body.z;

        bufferIndex = (bufferIndex + 1) % windowSize;
        if (filled < windowSize) filled++;

        if (filled == windowSize)
        {
            float[] flat = new float[windowSize * featuresPerReading];
            int idx = 0;
            int start = bufferIndex; // oldest index
            for (int w = 0; w < windowSize; w++)
            {
                int row = (start + w) % windowSize;
                for (int f = 0; f < featuresPerReading; f++)
                    flat[idx++] = buffer[row, f];
            }
            lastFlatWindow = (float[])flat.Clone();
            OnWindowReadyEvent?.Invoke(flat);
        }
    }
}
