// PIDController.cs
// Simple vector PID used by controllers.

using UnityEngine;

public class PIDController
{
    public float Kp, Ki, Kd;
    private Vector3 integral;
    private Vector3 prevError;
    private bool first = true;

    public PIDController(float kp, float ki, float kd)
    {
        Kp = kp; Ki = ki; Kd = kd;
        integral = Vector3.zero;
        prevError = Vector3.zero;
    }

    // step returns control vector (units depend on tuning)
    public Vector3 Step(Vector3 error, float dt)
    {
        if (first)
        {
            prevError = error;
            first = false;
        }
        integral += error * dt;
        Vector3 derivative = (error - prevError) / Mathf.Max(dt, 1e-6f);
        prevError = error;
        return Kp * error + Ki * integral + Kd * derivative;
    }

    public void Reset()
    {
        integral = Vector3.zero;
        prevError = Vector3.zero;
        first = true;
    }
}
