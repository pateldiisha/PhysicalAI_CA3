using UnityEngine;


[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    public float holdSeconds = 3f;

    // Gen3 joint ranges (radians); must match JOINT_LO / JOINT_HI in train_controller.py.
    // joint_2/4/6 come from the MJCF; the continuous joints 1/3/5/7 use home +/- pi.
    public Vector2[] jointRanges =
    {
        new Vector2(-3.14159265f, 3.14159265f),
        new Vector2(-2.24f, 2.24f),
        new Vector2(0f, 6.28318531f),
        new Vector2(-2.57f, 2.57f),
        new Vector2(-3.14159265f, 3.14159265f),
        new Vector2(-2.09f, 2.09f),
        new Vector2(-1.57079633f, 4.71238898f),
    };

    // Gen3 "home" keyframe; must match HOME in train_controller.py
    public float[] homeTargets = { 0f, 0.26179939f, 3.14159265f, -2.26892803f, 0f, 0.95993109f, 1.57079633f };

    [Range(0f, 1f)]
    public float randomScale = 0.35f;

    public bool emitActions = true;

    // When false, goals only change through SetTarget (used by IkGoalProvider)
    public bool resampleGoals = true;

    // Teleport the arm to homeTargets once the MuJoCo scene exists
    public bool startAtHome = true;

    float[] target;
    float hold;
    bool ready;
    bool placedAtHome;

    public float[] CurrentTarget => target;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("ArmJointController needs a ready ArmAgent.");
            enabled = false;
            return;
        }

        int n = agent.ActionSize;
        if (jointRanges == null || jointRanges.Length != n)
        {
            Debug.LogWarning("ArmJointController: jointRanges length != ActionSize, using [-pi, pi].");
            jointRanges = new Vector2[n];
            for (int i = 0; i < n; i++)
                jointRanges[i] = new Vector2(-Mathf.PI, Mathf.PI);
        }
        if (homeTargets == null || homeTargets.Length != n)
        {
            Debug.LogWarning("ArmJointController: homeTargets length != ActionSize, using zeros.");
            homeTargets = new float[n];
        }

        target = new float[n];
        PlaceTarget(true);
        hold = holdSeconds;
        agent.SetAction(target);
        ready = true;
    }

    void FixedUpdate()
    {
        if (!ready)
            return;

        if (startAtHome && !placedAtHome)
            placedAtHome = agent.SetJointState(homeTargets);

        if (resampleGoals)
        {
            hold -= Time.fixedDeltaTime;
            if (hold <= 0f)
            {
                PlaceTarget(false);
                hold = holdSeconds;
            }
        }

        if (emitActions)
            agent.SetAction(target);
    }

    public void ActionFrom(float[] destination)
    {
        int count = Mathf.Min(destination.Length, target.Length);
        for (int i = 0; i < count; i++)
            destination[i] = target[i];
    }

    public void PlaceTarget(bool home)
    {
        for (int i = 0; i < target.Length; i++)
        {
            float lo = jointRanges[i].x;
            float hi = jointRanges[i].y;
            float goal = homeTargets[i];
            if (!home)
            {
                // Same distribution as sample_goal() in train_controller.py
                float span = (hi - lo) * 0.5f * randomScale;
                goal += Random.Range(-span, span);
            }
            target[i] = Mathf.Clamp(goal, lo, hi);
        }
    }

    public void SetTarget(float[] goal)
    {
        if (target == null || goal == null)
            return;
        int count = Mathf.Min(goal.Length, target.Length);
        for (int i = 0; i < count; i++)
            target[i] = Mathf.Clamp(goal[i], jointRanges[i].x, jointRanges[i].y);
    }
}
