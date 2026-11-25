using System.IO;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>One-click import and wiring for the CA3 Kinova Gen3 scene.</summary>
[InitializeOnLoad]
public static class CA3SceneSetup
{
    const string ScenePath = "Assets/CA3/Scenes/Gen3Scene.unity";
    const string ModelPath = "Assets/CA3/kinova_gen3/scene.xml";
    const string AutoImportKey = "CA3.Gen3SceneImported";

    static CA3SceneSetup()
    {
        // Finish first-run setup after the scripts compile, so adding the project to Hub is enough
        // to produce a ready-to-inspect scene even when the native file picker is unavailable.
        EditorApplication.delayCall += () =>
        {
            if (EditorPrefs.GetBool(AutoImportKey, false) || File.Exists(ScenePath) || EditorApplication.isPlaying)
                return;
            try
            {
                ImportAndConfigure();
                if (File.Exists(ScenePath))
                    EditorPrefs.SetBool(AutoImportKey, true);
            }
            catch (System.Exception exception)
            {
                Debug.LogException(exception);
            }
        };
    }

    [MenuItem("CA3/Import and Configure Gen3 Scene")]
    public static void ImportAndConfigure()
    {
        if (MjScene.InstanceExists)
        {
            EditorUtility.DisplayDialog("MuJoCo scene already exists",
                "Close the current scene before importing another MuJoCo model. The package uses one MjScene per Unity scene.", "OK");
            return;
        }

        if (!File.Exists(ModelPath))
        {
            EditorUtility.DisplayDialog("Gen3 model not found", "Expected " + ModelPath, "OK");
            return;
        }

        var root = new MjImporterWithAssets().ImportFile(Path.GetFullPath(ModelPath));
        if (root == null)
        {
            Debug.LogError("MuJoCo importer returned no scene root.");
            return;
        }

        // Agent comes first so its Awake resolves the imported joints/actuators before consumers start.
        var agent = root.AddComponent<ArmAgent>();
        var goals = root.AddComponent<ArmJointController>();
        goals.resampleGoals = false;
        goals.emitActions = true;
        goals.startAtHome = true;

        // The recorder is ready for data collection but starts disabled so the user can inspect the IK scene first.
        var recorder = root.AddComponent<RolloutRecorder>();
        recorder.enabled = false;

        var controller = root.AddComponent<WorldModelController>();
        controller.enabled = false;

        var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        target.name = "TrackingTarget";
        Object.DestroyImmediate(target.GetComponent<Collider>());
        target.transform.position = root.transform.TransformPoint(new Vector3(0.5f, 0.2f, 0.45f));
        target.transform.localScale = Vector3.one * 0.05f;
        var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        var targetMaterial = new Material(shader) { color = new Color(1f, 0.35f, 0.05f, 1f) };
        target.GetComponent<Renderer>().sharedMaterial = targetMaterial;

        var ik = root.AddComponent<IkGoalProvider>();
        ik.agent = agent;
        ik.goals = goals;
        ik.target = target.transform;

        var camera = Camera.main;
        if (camera != null)
        {
            camera.transform.position = new Vector3(2.1f, 1.8f, -3f);
            camera.transform.LookAt(root.transform.position + Vector3.up * 0.45f);
            camera.fieldOfView = 45f;
        }

        Directory.CreateDirectory("Assets/CA3/Scenes");
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        var scene = SceneManager.GetActiveScene();
        EditorSceneManager.SaveScene(scene, ScenePath);
        Selection.activeGameObject = root;
        Debug.Log("Imported Kinova Gen3 and saved " + ScenePath + ". IK drives direct joint targets until trained model assets are available.");
    }
}
