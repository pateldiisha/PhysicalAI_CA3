using System;
using UnityEngine;


[DefaultExecutionOrder(-5)]
public class WorldModelController : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public TextAsset controllerJson;

    // Must match ACTION_SMOOTH / MAX_DELTA in train_controller.py
    [Range(0.0f, 1f)]
    public float actionSmoothing = 0.2f;

    public float maxDeltaPerStep = 0.05f;

    float[] W;
    float[] b;
    float[] jointLo;
    float[] jointHi;

    float[] features;
    float[] rawAction;
    float[] action;
    bool ready;
    bool hasAction;
    bool tookOverEmitActions;
    bool previousEmitActions;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || controllerJson == null)
        {
            Debug.LogError("WorldModelController needs agent, goals, and controller.json.");
            enabled = false;
            return;
        }

        features = new float[agent.StateSize];
        rawAction = new float[agent.ActionSize];
        action = new float[agent.ActionSize];

        if (!Load(controllerJson.text))
        {
            enabled = false;
            return;
        }

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        ready = true;
        Debug.Log(
            "WorldModelController ready (CMA-ES Controller C). " +
            "Goals come from ArmJointController; actions come from the learned linear policy.");
    }

    void OnEnable()
    {
        if (!ready || goals == null)
            return;

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        hasAction = false;
    }

    void OnDisable()
    {
        if (tookOverEmitActions && goals != null)
        {
            goals.emitActions = previousEmitActions;
            tookOverEmitActions = false;
        }
        hasAction = false;
    }

    void FixedUpdate()
    {
        if (!ready || !agent.Ready || goals.CurrentTarget == null)
            return;

        var state = agent.Observe();
        // The smoothing chain starts from the current joint angles
        if (!hasAction)
        {
            for (int i = 0; i < agent.ActionSize; i++)
                action[i] = state[i];
            hasAction = true;
        }

        ActionFrom(state, goals.CurrentTarget, action);
        agent.SetAction(action);
    }

    public void ActionFrom(float[] state, float[] destination)
    {
        ActionFrom(state, goals.CurrentTarget, destination);
    }

    // Linear policy on features [q - goal, qd]: raw = goal + W f + b.
    // destination holds the previous action on entry and is smoothed in place,
    // so each caller (this controller, the WorldModelDriver ghost) owns its own smoothing state.
    public void ActionFrom(float[] state, float[] goal, float[] destination)
    {
        int n = agent.ActionSize;
        for (int i = 0; i < n; i++)
        {
            features[i] = state[i] - goal[i];
            features[n + i] = state[n + i];
        }

        for (int r = 0; r < n; r++)
        {
            float sum = b[r];
            int row = r * agent.StateSize;
            for (int c = 0; c < agent.StateSize; c++)
                sum += W[row + c] * features[c];
            rawAction[r] = goal[r] + sum;
        }

        ApplySmoothing(rawAction, destination);
    }

    // Mirrors smooth_action() in train_controller.py
    void ApplySmoothing(float[] raw, float[] smoothed)
    {
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float prev = smoothed[i];
            float blended = (1f - actionSmoothing) * prev + actionSmoothing * raw[i];
            float delta = Mathf.Clamp(blended - prev, -maxDeltaPerStep, maxDeltaPerStep);
            smoothed[i] = Mathf.Clamp(prev + delta, jointLo[i], jointHi[i]);
        }
    }

    bool Load(string json)
    {
        try
        {
            var data = JsonUtility.FromJson<ControllerFile>(json);
            if (data == null || data.W == null || data.b == null ||
                data.W.Length != agent.ActionSize * agent.StateSize ||
                data.b.Length != agent.ActionSize)
            {
                Debug.LogError(
                    "controller.json has unexpected W/b shapes for ActionSize=" +
                    agent.ActionSize + " StateSize=" + agent.StateSize + ".");
                return false;
            }

            W = data.W;
            b = data.b;
            jointLo = data.joint_lo != null && data.joint_lo.Length == agent.ActionSize
                ? data.joint_lo
                : CopyRangesLo();
            jointHi = data.joint_hi != null && data.joint_hi.Length == agent.ActionSize
                ? data.joint_hi
                : CopyRangesHi();
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            return false;
        }
    }

    float[] CopyRangesLo()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].x
                : -Mathf.PI;
        return result;
    }

    float[] CopyRangesHi()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].y
                : Mathf.PI;
        return result;
    }

    [Serializable]
    class ControllerFile
    {
        public string feature;
        public int state_size;
        public int action_size;
        public float[] W;
        public float[] b;
        public float[] joint_lo;
        public float[] joint_hi;
        public float[] home;
    }
}
