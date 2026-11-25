# CA3: MuJoCo Arm World Model

This Unity project uses the MuJoCo 3.17.0 plugin and a seven-joint Kinova Gen3 arm. It records joint-space rollouts, trains a learned dynamics model and controller, and uses inverse kinematics to reach a movable target. See [the CA3 report](Assets/CA3/REPORT.md) and [setup notes](Assets/CA3/SETUP.md) for project details.

## AI Assistance

I used Claude as a coding assistant to help implement and debug some parts of the MuJoCo-to-Unity workflow, including rollout recording, world-model training, the learned controller, and inverse kinematics. I also used it to review project setup and verification results and to organize this README. I implemented and reviewed the code and metrics to describe how the system works. I wrote the reflection according to my understanding of this assignment.

## Reflection

1. **What were the hardest part for you, and what part was the easiest? How did you overcome this difficulty?**  
   The hardest part was integrating the imported Gen3 with Unity: the wrist camera covered the Game view, and recording each action with the correct before-and-after MuJoCo states took careful timing; I fixed the camera setup in `ArmAgent` and used `RolloutRecorder` to capture transitions after each physics step, while writing the q, qd, and target columns became straightforward once I identified the seven joints.

2. **How does your World Model work? What data is collected, what is your model architecture, what does your model predict?**  
   My World Model trains on 14,151 recorded transitions, with each input containing the Gen3’s seven joint angles, seven velocities, and seven commanded joint targets; a normalized 21→256→256→256→14 SiLU MLP predicts the changes in angles and velocities for the next step.

3. **How does your controller work? How is it trained, how does it connect to your world model, what does it control?**  
   The controller is a linear policy trained with CMA-ES over imagined rollouts from that model; its inputs are joint-position error and velocity, and its smoothed, range-limited joint-target outputs tell the Gen3 how to track a goal.

4. **What does your IK solver do? How does it work and how does it connect to your controller? Why and how did you pick this algorithm?**  
   My IK solver uses damped least squares with the MuJoCo Jacobian for `bracelet_link`, which I chose for stable position-based reaching with the Gen3’s seven degrees of freedom; it adds a small pull toward the home pose, then passes the solved joint targets to the controller.

5. **How has your understanding of World Models changed while doing this assignment?**  
   I used to think a World Model was mainly a representation of the scene, but this assignment taught me to treat it as a learned predictor that lets a controller test actions before taking them; I also saw why prediction errors matter, since held-out joint RMSE rises from 0.06083 radians for one step to 0.23343 over 100 imagined steps.

6. **How might you implement what you have learned here into your Term Project?**  
   For my Term Project’s Digital Roommate, I could apply that lesson by predicting how actions affect the cup’s location and status, then comparing those predictions with the observed world state so the behavior tree can replan when someone moves the cup while the roommate is away.
