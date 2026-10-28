# Real-to-sim plan

## 1. Definition
Real-to-sim makes a simulation of a real scene from a capture of that scene.
LiveSplat input: one phone video of a tabletop.
LiveSplat output: a **sim package**. A robotics team loads the package in a simulator and uses it immediately.

## 2. Problem
Robot teams test manipulation in simulation before they use the real robot.
To copy a real table into a simulator, a person must:
1. Model each object (mesh).
2. Make collision shapes.
3. Set mass, friction and inertia.
4. Place each object at the correct pose.
5. Match the appearance, so that camera-based policies see the same images.
This work takes hours to days for each scene. The result is often not similar to the real scene ("sim-to-real gap").
Research tools (SplatSim, RoboGSim, phi-RIE) show the value. They need manual setup.

## 3. What LiveSplat does
| Step | Output | Status |
|---|---|---|
| Video to splat | Photoreal 3D scene | Done (ThriftSplat) |
| Table and objects | Separate objects, table plane, Z up | Done |
| Collision shapes | Convex hull per object, flat footprint | Done |
| Physical properties | Mass, friction, bounce (Spark estimate) | Done (estimate) |
| Names | Object names and aliases (Spark) | Done |
| **Export package** | MJCF + URDF + meshes + splats + metadata | **New** |
| **Photoreal sim render** | Images of the sim state from any camera, with splats | **New** |
| **Robot check** | Reach and top-down grasp test with a Franka Panda | **New** |
| **Variants** | N randomized scenes (pose, mass, friction, colour, scale) | **New** |
| Metric scale | Reference object | Needs a new capture |

## 4. Sim package contents
```
<name>_sim/
  scene.xml            MJCF: table, floor, objects (free joints), capture cameras
  robot_scene.xml      scene.xml + Franka Panda at the table edge (if robot check ran)
  urdf/obj_<k>.urdf    one URDF per object (for PyBullet, Isaac, Gazebo users)
  meshes/obj_<k>_collision.obj   convex collision mesh (table frame, metres*)
  meshes/obj_<k>_visual.obj      simple visual mesh, rgba = median object colour
  meshes/table.obj
  splats/obj_<k>.ply   object splat in its body frame (3DGS PLY, raw values)
  splats/background.ply
  metadata.json        objects, poses, mass, friction, sizes, cameras, frames, units, provenance, uncertainty
  render.py            renders the sim state with splats (gsplat)
  README.md            how to load the package
```
*Scale is "scene units" until a reference object gives metres. metadata.json states this.

## 5. Tasks and needs
| Task | Needs | Risk |
|---|---|---|
| T1 Export MJCF/URDF/meshes/metadata | trimesh, current hulls, props.json | Low |
| T2 Package self-test | Load scene.xml in plain MuJoCo, settle 2 s, objects stay | Low |
| T3 Photoreal render from sim state | gsplat in the LiveSplat venv; splat per body, transform by body pose | Medium (gsplat compile) |
| T4 Render check | Render the capture cameras at rest; compare with keyframes (PSNR) | Low |
| T5 Robot model | Franka Panda MJCF (MuJoCo Menagerie, Apache-2.0) | Low (one download) |
| T6 Reach check | IK to a point above each object; report reachable or not | Medium |
| T7 Grasp check | Object width at grasp height vs gripper opening (8 cm); top-down grasp + lift in sim; report success | Medium-high |
| T8 Variants | Randomize pose/mass/friction/colour/scale with a seed; export N packages | Low |
| T9 Viewer | "Export sim" and "robot check" buttons; show the robot | Low |
| T10 Concave objects | Convex decomposition (CoACD) for mugs, bowls | Optional |

## 6. Value for robotics teams and labs
1. **Scene setup in minutes, not days.** One video gives a ready MJCF scene.
2. **Photoreal camera images.** Vision policies see images close to the real camera (splat render).
3. **Pre-check before real trials.** Reach and grasp checks show which objects the robot can handle.
4. **Domain randomization.** Variants give many training or test scenes from one capture.
5. **Standard formats.** MJCF and URDF load in MuJoCo, PyBullet, Isaac, Gazebo.
6. **Honest metadata.** Each value has its source (measured, estimated, default) and an uncertainty.

## 7. Limits (state these in the demo)
- Scale is not metric without a reference object.
- Mass and friction are estimates (Spark vision). Example: toilet roll 0.02 kg is wrong.
- Collision shapes are convex. Concave objects (mug inside) need T10.
- Touching objects become one object.
- Undersides of objects are not seen. The footprint is flat.

## 8. Order of work
1. T1 + T2 (export + self-test). Test: package loads and settles.
2. T3 + T4 (render). Test: PSNR of the render at a capture camera.
3. T5 + T6 (robot + reach). Test: reach result for each object.
4. T8 (variants). Test: N packages, all load and settle.
5. T7 (grasp). Test: top-down grasp + lift on table1 objects.
6. T9 (viewer buttons).
7. T10 only if time remains.

## 9. Rules
- No paid API calls in this work (CLAUDE.md rule 1).
- Do not copy ThriftSplat code. Import it if necessary.
- Each task ends with a test and a commit.
