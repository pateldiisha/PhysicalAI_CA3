using Mujoco;
using UnityEngine;

// Runs before MjScene (order 0) so the control written here is used by this step's mj_step.
[DefaultExecutionOrder(-1)]
public class ArmAgent : MonoBehaviour
{
    public int JointCount = 7;
    public int StateSize = 14;
    public int ActionSize = 7;

    // MJCF names (defaults are for Menagerie kinova_gen3; actuators share the joint names)
    public string[] jointNames = { "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7" };
    public string[] actuatorNames = { "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7" };
    public string endEffectorName = "bracelet_link";
    // Gen3 has no gripper; leave empty unless your arm has one
    public string gripperName = "";
    public float gripperControl = 0f;

    public MjActuator gripper;
    public MjBody endEffectorBody;
    public Transform endEffector;
    public MjHingeJoint[] joints;
    public MjActuator[] actuators;

    float[] action;
    float[] state;
    bool ready;
    bool hasAction;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    void Awake()
    {
        StateSize = 2 * JointCount;
        ActionSize = JointCount;
        action = new float[ActionSize];
        state = new float[StateSize];
        InitializeJointsAndActuators();
        ConfigurePresentationCamera();
    }

    void ConfigurePresentationCamera()
    {
        // The imported Gen3 wrist sensor is represented as a Unity Camera. If
        // left enabled it can render after Main Camera and replace the whole
        // Game view with its sensor view. The assignment uses the scene camera.
        var cameras = FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        foreach (var camera in cameras)
            if (camera != null && camera.gameObject != null && camera.gameObject.tag != "MainCamera")
                camera.enabled = false;

        var sceneCamera = Camera.main;
        if (sceneCamera == null)
            return;

        // Keep the full arm in frame, independent of the importer sensor
        // camera's orientation and the editor's saved camera view.
        sceneCamera.transform.position = transform.position + new Vector3(1.4f, 1.2f, -2f);
        sceneCamera.transform.LookAt(transform.position + new Vector3(0f, 0.35f, 0f));
        sceneCamera.nearClipPlane = 0.05f;
        sceneCamera.farClipPlane = 100f;
    }

    void FixedUpdate()
    {
        if (!ready || !hasAction)
            return;
        ApplyAction();
    }

    public void InitializeJointsAndActuators()
    {
        // The importer leaves Unity renderers disabled in its saved scene. Enable
        // the generated meshes so the simulated robot is visible to Unity cameras.
        foreach (var geom in GetComponentsInChildren<MjGeom>(true))
        {
            var renderer = geom.GetComponent<MeshRenderer>();
            if (renderer != null)
                renderer.enabled = true;
        }

        if (joints == null || joints.Length != JointCount)
            joints = new MjHingeJoint[JointCount];
        if (actuators == null || actuators.Length != ActionSize)
            actuators = new MjActuator[ActionSize];

        for (int i = 0; i < JointCount; i++)
            if (joints[i] == null && i < jointNames.Length)
                joints[i] = FindComponent<MjHingeJoint>(jointNames[i]);
        for (int i = 0; i < ActionSize; i++)
            if (actuators[i] == null && i < actuatorNames.Length)
                actuators[i] = FindComponent<MjActuator>(actuatorNames[i]);

        if (endEffectorBody == null)
            endEffectorBody = FindComponent<MjBody>(endEffectorName);
        if (endEffector == null && endEffectorBody != null)
            endEffector = endEffectorBody.transform;
        if (gripper == null && !string.IsNullOrEmpty(gripperName))
            gripper = FindComponent<MjActuator>(gripperName);

        ready = endEffectorBody != null;
        for (int i = 0; i < JointCount; i++)
        {
            if (joints[i] == null)
            {
                Debug.LogError("ArmAgent: joint '" + jointNames[i] + "' not found.");
                ready = false;
            }
        }
        for (int i = 0; i < ActionSize; i++)
        {
            if (actuators[i] == null)
            {
                Debug.LogError("ArmAgent: actuator '" + actuatorNames[i] + "' not found.");
                ready = false;
            }
        }
        if (endEffectorBody == null)
            Debug.LogError("ArmAgent: end effector body '" + endEffectorName + "' not found.");
    }

    public void SetAction(float[] targets)
    {
        if (targets == null)
            return;
        int count = Mathf.Min(targets.Length, ActionSize);
        for (int i = 0; i < count; i++)
            action[i] = targets[i];
        hasAction = true;
    }

    public unsafe void ApplyAction()
    {
        bool live = MjScene.InstanceExists && MjScene.Instance.Data != null;
        var data = live ? MjScene.Instance.Data : null;

        for (int i = 0; i < ActionSize; i++)
        {
            actuators[i].Control = action[i];
            // MjScene only syncs Control after the step, so write ctrl directly to avoid a one-step lag
            if (live && actuators[i].MujocoId >= 0)
                data->ctrl[actuators[i].MujocoId] = action[i];
        }

        if (gripper != null)
        {
            gripper.Control = gripperControl;
            if (live && gripper.MujocoId >= 0)
                data->ctrl[gripper.MujocoId] = gripperControl;
        }
    }

    // State layout: [q_0..q_{n-1}, qd_0..qd_{n-1}] in radians and rad/s
    public unsafe float[] Observe()
    {
        if (!ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return state;

        var data = MjScene.Instance.Data;
        for (int i = 0; i < JointCount; i++)
        {
            state[i] = (float)data->qpos[joints[i].QposAddress];
            state[JointCount + i] = (float)data->qvel[joints[i].DofAddress];
        }
        return state;
    }

    public void CopyState(float[] destination)
    {
        var current = Observe();
        int count = Mathf.Min(destination.Length, current.Length);
        for (int i = 0; i < count; i++)
            destination[i] = current[i];
    }

    // Teleports the joints to q with zero velocity. Returns false until the MuJoCo scene exists.
    public unsafe bool SetJointState(float[] q)
    {
        if (!ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return false;

        var scene = MjScene.Instance;
        var data = scene.Data;
        for (int i = 0; i < JointCount; i++)
        {
            data->qpos[joints[i].QposAddress] = q[i];
            data->qvel[joints[i].DofAddress] = 0;
        }
        MujocoLib.mj_forward(scene.Model, data);
        return true;
    }

    private T FindComponent<T>(string name) where T : Component
    {
        if (string.IsNullOrEmpty(name))
            return null;

        // Prefer components under this arm, then fall back to the whole scene
        var local = GetComponentsInChildren<T>(true);
        var found = MatchName(local, name);
        if (found != null)
            return found;

        var all = FindObjectsByType<T>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        return MatchName(all, name);
    }

    static T MatchName<T>(T[] candidates, string name) where T : Component
    {
        foreach (var candidate in candidates)
            if (candidate.name == name)
                return candidate;
        // Imported names are sometimes prefixed (e.g. "panda/joint1" or "panda_joint1")
        foreach (var candidate in candidates)
            if (candidate.name.EndsWith("/" + name) || candidate.name.EndsWith("_" + name))
                return candidate;
        return null;
    }
}
