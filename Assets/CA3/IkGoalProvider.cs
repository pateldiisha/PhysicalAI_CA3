using Mujoco;
using UnityEngine;

// Cartesian IK that produces joint goals for the learned controller.
// Damped least squares (Levenberg-Marquardt) on the MuJoCo body Jacobian, with a null-space
// pull toward the home pose to resolve the 7-DOF redundancy.
// Runs after ArmJointController (-10) and before WorldModelController (-5).
[DefaultExecutionOrder(-7)]
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public Transform target;

    public int iterations = 20;
    public float damping = 0.05f;
    public float maxStep = 0.2f;
    public float tolerance = 0.002f;
    [Range(0f, 1f)]
    public float nullSpaceGain = 0.1f;

    // Distance from the end effector to the target at the solved pose (metres)
    public float residual;
    public bool reachable;

    float[] q;
    double[] savedQ;
    double[] jacp;
    float[] jac;      // 3 x n, columns are the arm dofs
    bool solved;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || target == null)
        {
            Debug.LogError("IkGoalProvider needs agent, goals (ArmJointController), and a target Transform.");
            enabled = false;
            return;
        }

        int n = agent.JointCount;
        q = new float[n];
        savedQ = new double[n];
        jac = new float[3 * n];
    }

    void OnEnable()
    {
        if (goals != null)
            goals.resampleGoals = false;
        solved = false;
    }

    void OnDisable()
    {
        if (goals != null)
            goals.resampleGoals = true;
    }

    unsafe void FixedUpdate()
    {
        if (!agent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        var scene = MjScene.Instance;
        var model = scene.Model;
        var data = scene.Data;
        int n = agent.JointCount;
        int nv = (int)model->nv;

        if (jacp == null || jacp.Length != 3 * nv)
            jacp = new double[3 * nv];

        for (int i = 0; i < n; i++)
            savedQ[i] = data->qpos[agent.joints[i].QposAddress];

        // Warm start from the previous solution so the goal moves continuously with the target
        if (!solved)
            for (int i = 0; i < n; i++)
                q[i] = (float)savedQ[i];

        Vector3 goal = MjVector(target.position);
        int body = agent.endEffectorBody.MujocoId;

        for (int it = 0; it < iterations; it++)
        {
            for (int i = 0; i < n; i++)
                data->qpos[agent.joints[i].QposAddress] = q[i];
            MujocoLib.mj_kinematics(model, data);

            double* p = data->xpos + 3 * body;
            Vector3 err = goal - new Vector3((float)p[0], (float)p[1], (float)p[2]);
            residual = err.magnitude;
            if (residual < tolerance)
                break;

            fixed (double* jp = jacp)
                MujocoLib.mj_jacBody(model, data, jp, null, body);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < n; c++)
                    jac[r * n + c] = (float)jacp[r * nv + agent.joints[c].DofAddress];

            Step(err, n);
        }

        // Final residual at the solution
        for (int i = 0; i < n; i++)
            data->qpos[agent.joints[i].QposAddress] = q[i];
        MujocoLib.mj_kinematics(model, data);
        double* ee = data->xpos + 3 * body;
        residual = (goal - new Vector3((float)ee[0], (float)ee[1], (float)ee[2])).magnitude;
        reachable = residual < 0.01f;

        // Restore the real arm's state
        for (int i = 0; i < n; i++)
            data->qpos[agent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(model, data);

        goals.SetTarget(q);
        // SetTarget clamps to the joint ranges; keep the warm start consistent with that
        System.Array.Copy(goals.CurrentTarget, q, n);
        solved = true;
    }

    // dq = J^T (J J^T + l^2 I)^-1 e  +  (I - J^+ J) k (home - q)
    void Step(Vector3 err, int n)
    {
        var a = new Matrix4x4();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                float sum = 0f;
                for (int k = 0; k < n; k++)
                    sum += jac[r * n + k] * jac[c * n + k];
                a[r, c] = sum + (r == c ? damping * damping : 0f);
            }
        a[3, 3] = 1f;
        var aInv = a.inverse;

        // Primary task: y = A^-1 e, dq = J^T y
        Vector3 y = aInv.MultiplyVector(err);
        var dq = new float[n];
        for (int k = 0; k < n; k++)
            dq[k] = jac[k] * y.x + jac[n + k] * y.y + jac[2 * n + k] * y.z;

        // Secondary task projected into the null space: z - J^T A^-1 J z
        if (nullSpaceGain > 0f)
        {
            var z = new float[n];
            for (int k = 0; k < n; k++)
                z[k] = nullSpaceGain * (goals.homeTargets[k] - q[k]);
            Vector3 jz = Vector3.zero;
            for (int k = 0; k < n; k++)
            {
                jz.x += jac[k] * z[k];
                jz.y += jac[n + k] * z[k];
                jz.z += jac[2 * n + k] * z[k];
            }
            Vector3 w = aInv.MultiplyVector(jz);
            for (int k = 0; k < n; k++)
                dq[k] += z[k] - (jac[k] * w.x + jac[n + k] * w.y + jac[2 * n + k] * w.z);
        }

        float norm = 0f;
        for (int k = 0; k < n; k++)
            norm += dq[k] * dq[k];
        norm = Mathf.Sqrt(norm);
        float scale = norm > maxStep ? maxStep / norm : 1f;

        for (int k = 0; k < n; k++)
        {
            var range = goals.jointRanges[k];
            q[k] = Mathf.Clamp(q[k] + scale * dq[k], range.x, range.y);
        }
    }

    // Unity (x, y, z) -> MuJoCo (x, z, y), the inverse of MjEngineTool.UnityVector3
    static Vector3 MjVector(Vector3 unity) => new Vector3(unity.x, unity.z, unity.y);

    void OnDrawGizmos()
    {
        if (target == null)
            return;
        Gizmos.color = reachable ? Color.green : Color.red;
        Gizmos.DrawWireSphere(target.position, 0.03f);
    }
}
