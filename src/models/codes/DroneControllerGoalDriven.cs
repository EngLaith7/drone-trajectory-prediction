// DroneControllerGoalDriven.cs
// Closed-loop controller: uses IMU (SimulatedIMU or IMUFromCSV) + RF model to generate next-step predictions
// and blends model prediction with a user-provided goal position. Controller uses PD-like control.

using System.Collections;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class DroneControllerGoalDriven : MonoBehaviour
{
    [Header("Model")]
    public string modelJsonFileName = "rf_model.json";
    public bool requireModel = true;

    [Header("Goal")]
    public Transform goalTransform; // user-provided end point
    [Range(0f,1f)] public float modelTrust = 0.6f; // 0..1 blend model vs goal

    [Header("Control")]
    public float controlInterval = 0.02f;
    public float maxVel = 3f;
    public float Kp = 1.2f, Kd = 0.2f;
    public float arrivalThreshold = 0.2f;

    [Header("Debug")]
    public bool log = false;

    private RFModelRuntime rf;
    private SimulatedIMU simImu;
    private IMUFromCSV csvImu;
    private Rigidbody rb;
    private Vector3 prevPosError = Vector3.zero;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        // Attempt to load model
        try
        {
            rf = new RFModelRuntime(modelJsonFileName);
        }
        catch (System.Exception ex)
        {
            rf = null;
            Debug.LogWarning("[DroneControllerGoalDriven] Model load failed: " + ex.Message);
            if (requireModel) enabled = false;
        }

        // find IMU source on same GameObject
        simImu = GetComponent<SimulatedIMU>();
        csvImu = GetComponent<IMUFromCSV>();
        if (simImu == null && csvImu == null) Debug.LogWarning("[DroneControllerGoalDriven] No IMU source attached (SimulatedIMU or IMUFromCSV).");

        StartCoroutine(ControlLoop());
    }

    IEnumerator ControlLoop()
    {
        while (true)
        {
            float t0 = Time.time;

            if (goalTransform == null)
            {
                if (log) Debug.Log("[DroneControllerGoalDriven] No goal assigned.");
                yield return new WaitForSeconds(controlInterval);
                continue;
            }

            // choose available IMU and get latest flat window
            float[] flat = null;
            if (simImu != null) flat = simImu.GetLatestFlatWindow();
            if (flat == null && csvImu != null) flat = csvImu.GetLatestFlatWindow();

            float[] pred = null;
            if (flat != null && rf != null)
            {
                try { pred = rf.Predict(flat); }
                catch (System.Exception ex) { Debug.LogError("[DroneControllerGoalDriven] Predict error: " + ex.Message); }
            }

            Vector3 modelPos = pred != null ? new Vector3(pred[0], pred[1], pred[2]) : rb.position;
            Vector3 goalPos = goalTransform.position;

            // compute commanded target: blend goal and model prediction
            Vector3 commandedTarget = Vector3.Lerp(goalPos, modelPos, modelTrust);

            // PD-like control towards commandedTarget
            Vector3 posError = commandedTarget - rb.position;
            Vector3 derivative = (posError - prevPosError) / Mathf.Max(controlInterval, 1e-6f);
            prevPosError = posError;

            Vector3 controlVel = Kp * posError + Kd * derivative;
            if (controlVel.magnitude > maxVel) controlVel = controlVel.normalized * maxVel;

            Vector3 nextPos = rb.position + controlVel * controlInterval;
            rb.MovePosition(nextPos);

            if (log)
            {
                Debug.Log($"pos={rb.position:F3} cmdTarget={commandedTarget:F3} modelPos={modelPos:F3} goal={goalPos:F3}");
            }

            // arrival check
            if (Vector3.Distance(rb.position, goalPos) < arrivalThreshold)
            {
                if (log) Debug.Log("[DroneControllerGoalDriven] Arrived at goal.");
                // optionally break or continue to hold position
            }

            float elapsed = Time.time - t0;
            float wait = Mathf.Max(0f, controlInterval - elapsed);
            yield return new WaitForSeconds(wait);
        }
    }
}
