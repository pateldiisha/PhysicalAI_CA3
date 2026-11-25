using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;

// Runs after MjScene (order 0), so Observe() sees the state *after* this step's mj_step.
[DefaultExecutionOrder(100)]
public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;

    // Relative to the Unity project root; must be the Data folder next to train.py
    public string dataFolder = "Assets/CA3/Data";
    // Skip the first few steps while the arm settles after the home teleport
    public int warmupSteps = 10;
    public int logEvery = 5000;

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    int warmup;
    float[] prevS;
    float[] prevA;
    bool saved;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("RolloutRecorder needs a ready ArmAgent.");
            enabled = false;
            return;
        }

        prevA = new float[agent.ActionSize];
        Debug.Log("RolloutRecorder: collecting " + steps + " transitions.");
    }

    unsafe void FixedUpdate()
    {
        if (saved || !agent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        // s is the post-step state; CurrentAction is the control that produced it
        var s = (float[])agent.Observe().Clone();
        var a = agent.CurrentAction;
        for (int i = 0; i < prevA.Length; i++)
            prevA[i] = a[i];

        if (warmup < warmupSteps)
        {
            warmup++;
            prevS = s;
            return;
        }

        if (prevS != null)
        {
            states.Add(prevS);
            actions.Add((float[])prevA.Clone());
            nextStates.Add(s);
            n++;

            if (logEvery > 0 && n % logEvery == 0)
                Debug.Log($"RolloutRecorder: {n}/{steps}  tracking RMSE {TrackingError():F3} rad");
        }
        prevS = s;

        if (n >= steps)
            Save();
    }

    // RMS joint error between the current state and the controller's goal
    float TrackingError()
    {
        if (prevS == null || controller == null || controller.CurrentTarget == null)
            return 0f;
        var goal = controller.CurrentTarget;
        float sum = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float e = prevS[i] - goal[i];
            sum += e * e;
        }
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    void OnDisable()
    {
        // Keep partial data if Play is stopped early
        if (!saved && n > 0)
            Save();
    }

    void Save()
    {
        if (saved)
            return;
        saved = true;

        string dir = Path.Combine(Path.GetDirectoryName(Application.dataPath), dataFolder);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "rollout.csv");

        var sb = new StringBuilder(n * 256);
        for (int row = 0; row < n; row++)
        {
            AppendRow(sb, states[row]);
            sb.Append(',');
            AppendRow(sb, actions[row]);
            sb.Append(',');
            AppendRow(sb, nextStates[row]);
            sb.Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"RolloutRecorder: wrote {n} transitions to {path}");
    }

    static void AppendRow(StringBuilder sb, float[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0)
                sb.Append(',');
            sb.Append(values[i].ToString("G9", CultureInfo.InvariantCulture));
        }
    }
}
