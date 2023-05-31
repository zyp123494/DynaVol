# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import os
# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-0.85, -0.85, 0.8), (0.85, 0.85, 1.1)]
#the region to place a table[(min), (max)], the third parameter is useless.
TABLE_REGION = [(-1, -1, 0.8), (1, 1, 1.1)]
VELOCITY_RANGE = [(-0.5, -0.5, 0.), (0.5, 0.5, 0.)]
#object scale
MAX_OBJECT_SCALE = 0.6
MIN_OBJECT_SCALE = 0.25

# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument("--num_dyn_objects", type=int, default=3,
                    help="number of dynamic objects")                  
parser.add_argument("--num_stc_objects", type=int, default=0,
                    help="number of static objects")
parser.add_argument("--xy_vel",  action = "store_true",
                    help="whether to add velocity along x and y axis")
parser.add_argument("--real_texture",  action = "store_true",
                    help="whether to use real-world texture")
parser.add_argument("--material", choices=["rubber", "metal"],
                    default="rubber")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored"],
                    default="clevr")
parser.add_argument("--job_dir", type=str, default="./output")


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=60, frame_rate=30,
                    resolution=512)
FLAGS = parser.parse_args()

def interpolate_camera_pos(start_point, num_frms, R = 4,end_z = 0.1):
    #return [num_frms,3] 
    #camera moves from start point with a circle with center = center
    
    start_z = start_point[-1]
    z = np.linspace(start_z, end_z, num_frms)
    R_t = np.sqrt(np.abs(R**2 - z**2))

    x0,y0 = start_point[:2]
    cx,cy = 0,0
    t0 = np.arccos((x0-cx)/R_t[0])
    t = np.linspace(t0,t0+4*np.pi,num_frms)
    x = cx + R_t*np.cos(t)
    y = cy + R_t*np.sin(t)
    pos = np.stack([x,y,z],axis =1)
    return pos
    

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
scene.gravity = (0,0,-1)

print(output_dir)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)

def _get_floor_scale_position_kwargs(
    spawn_region,
):
  """Constructs scale, position of the cube representing the floor.

  Cube's scale and position are chosen such that the cube's top surface lies
  strictly inside of spawn_region.

  Args:
    spawn_region: Region of the floor.

  Returns:
    {'scale': <XYZ multipliers for [-1, 1] cube>,
     'position': <XYZ center of floor cube>}
  """
  spawn_min, spawn_max = np.array(spawn_region)

  # Center of spawn position.
  position = (spawn_max + spawn_min) / 2

  # Set center of floor to be equal to bottom of spawn region.
  position[-1] = 0
  # Default cube has coordinates [[-1, -1, -1], [1, 1, 1]], so default cube
  # length is 2. So we scale floor by 2.
  scale = (spawn_max - spawn_min) / 2
  scale[0]+=0.1
  scale[1]+=0.1
  scale[-1] = 0.01  # Floor height is minimal.
  print(position,scale)
  return dict(
      position=tuple(position),
      scale=tuple(scale),
  )
# --- Populate the scene
# Floor / Background
logging.info("Creating a large cube as the floor...")
# Create the "floor" of the scene. This is a large, flat cube of
# x_length=200, y_length=200 and z_length=2 centered at x=0, y=0, z=-1.
# This means that the floor's elevation is z + z_length/2 == -1 + 1 == 0.

floor_material = kb.PrincipledBSDFMaterial(
    color=kb.Color.from_name("white"), roughness=1., specular=0.)
floor = kb.Cube(
    **_get_floor_scale_position_kwargs(TABLE_REGION),
    material=floor_material,
    friction=FLAGS.floor_friction,
    static=True,
    background=False)
floor.segmentation_id = 1
scene.add(floor)

background_color = kb.Color(0,0,0,0)
logging.info("Setting background color to %s...", background_color)
scene.background = background_color

# Lights
logging.info("Adding four (studio) lights to the scene similar to CLEVR...")
scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

gso = kb.AssetSource.from_manifest("gs://kubric-public/assets/GSO/GSO.json")
train_split, test_split = gso.get_test_split(fraction=0.1)

logging.info("Choosing one of the %d training objects...", len(train_split))
active_split = train_split



# add STATIC objects
num_static_objects = int(FLAGS.num_stc_objects)
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
    size_label = 'small'
    size = rng.uniform() * (MAX_OBJECT_SCALE - MIN_OBJECT_SCALE) + MIN_OBJECT_SCALE
    color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)


    material_name = FLAGS.material

    obj = gso.create(asset_id=rng.choice(active_split))
    assert isinstance(obj, kb.FileBasedObject)
    obj.scale = size/ np.max(obj.bounds[1] - obj.bounds[0])

    if not FLAGS.real_texture:
        if material_name == "metal":
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                                roughness=0.2, ior=2.5)
            obj.friction = 0.4
            obj.restitution = 0.3
            obj.mass *= 2.7 * size**3
        else:  
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                                ior=1.25, roughness=0.7,
                                                  specular=0.33)
            obj.friction = 0.8
            obj.restitution = 0.7
            obj.mass *= 1.1 * size**3
   
        obj.metadata = {
        "size": size,
        "size_label": size_label,
        "material": material_name.lower(),
        "color": random_color.rgb,
        "color_label": color_label,
        }
    obj.segmentation_id = i + 2
    obj.metadata["is_dynamic"] = False
    scene += obj
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION,
                            rng=rng)
    logging.info("    Added %s at %s", obj.asset_id, obj.position)


logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)


# stop any objects that are still moving and reset friction / restitution
for obj in scene.foreground_assets:
    if hasattr(obj, "velocity"):
        obj.velocity = (0., 0., 0.)
        obj.friction = 0.5
        obj.restitution = 0.5


# Add random objects
num_objects = int(FLAGS.num_dyn_objects)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
    size_label = 'small'
    size = rng.uniform() * (MAX_OBJECT_SCALE - MIN_OBJECT_SCALE) + MIN_OBJECT_SCALE
    color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)

    material_name = FLAGS.material

    obj = gso.create(asset_id=rng.choice(active_split))
    assert isinstance(obj, kb.FileBasedObject)
    obj.scale = size/ np.max(obj.bounds[1] - obj.bounds[0])

    if not FLAGS.real_texture:
        if material_name == "metal":
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                                    roughness=0.2, ior=2.5)
            obj.friction = 0.4
            obj.restitution = 0.3
            obj.mass *= 2.7 * size**3
        else:  # material_name == "rubber"
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                                    ior=1.25, roughness=0.7,
                                                    specular=0.33)
            obj.friction = 0.8
            obj.restitution = 0.7
            obj.mass *= 1.1 * size**3
  
        obj.metadata = {
        "size": size,
        "size_label": size_label,
        "material": material_name.lower(),
        "color": random_color.rgb,
        "color_label": color_label,
        }
    obj.segmentation_id = i + 2 + num_static_objects
    scene.add(obj)
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
    #initialize velocity randomly but biased towards center
    if FLAGS.xy_vel:
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
            [obj.position[0], obj.position[1], 0])
    else:
        obj.velocity = (0,0,0)

    logging.info("    Added %s at %s", obj.asset_id, obj.position)


if FLAGS.save_state:
    logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
    simulator.save_state(output_dir / "scene.bullet")



#render static dataset
split = ['train','val','test']


for dataset in split:
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=50., sensor_width=36)
        
    frames = []
    for (i,frame) in enumerate(range(FLAGS.frame_start, FLAGS.frame_end + 1)):
        
        print(frame)
        position = rng.normal(size=(3, ))
        position *= 4 / np.linalg.norm(position)
        position[2] = np.abs(position[2])
        scene.camera.position = position
        scene.camera.look_at((0, 0, 0))

        matrix = scene.camera.matrix_world
        frame = renderer.render_still()

        frame["segmentation"] = kb.adjust_segmentation_idxs(frame["segmentation"], scene.assets, [])
        
        kb.write_png(filename=output_dir /"static" /  dataset / f"{str(i).zfill(3)}.png", data=frame["rgba"])
        kb.write_palette_png(filename=output_dir / "static"/dataset / f"segmentation_{str(i).zfill(5)}.png", data=frame["segmentation"])

        frame_data = {
          "transform_matrix": matrix.tolist(),
          "file_path": f"{dataset}/{str(i).zfill(3)}",
        }
        frames.append(frame_data)

  # --- Write the JSON descriptor for this split
    kb.write_json(filename=output_dir /"static" / f"transforms_{dataset}.json", data={
      "camera_angle_x": scene.camera.field_of_view,
      "frames": frames,
    })




logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0,
                                            frame_end=scene.frame_end+1)

for dataset in split:
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=50., sensor_width=36)
        
    frames = []
    for (i,frame) in enumerate(range(FLAGS.frame_start - 1, FLAGS.frame_end + 2)):
        
        print(frame)
      
        position = rng.normal(size=(3, ))
        position *= 4 / np.linalg.norm(position)
        position[2] = np.abs(position[2])

        scene.camera.position = position
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

        matrix = scene.camera.matrix_world

        if frame >=FLAGS.frame_start and frame <= FLAGS.frame_end:
            time = (frame - FLAGS.frame_start) / (FLAGS.frame_end - FLAGS.frame_start)

            frame_data = {
              "transform_matrix": matrix.tolist(),
              "file_path": f"{dataset}/{str(frame-FLAGS.frame_start).zfill(3)}",
              "time": time,
            }
            frames.append(frame_data)
    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

   
    
    del data_stack["forward_flow"]
    del data_stack["backward_flow"]
    del data_stack["object_coordinates"]
    del data_stack["normal"]
    del data_stack["depth"]

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(data_stack["segmentation"], scene.assets, [])
    
    templates = {"rgba":"{:03d}.png"}

    # Save to image files
    kb.write_image_dict(data_stack, f'{output_dir}/dynamic/{dataset}',file_templates = templates)
    

  # --- Write the JSON descriptor for this split
    kb.write_json(filename=output_dir /"dynamic"/ f"transforms_{dataset}.json", data={
      "camera_angle_x": scene.camera.field_of_view,
      "frames": frames,
    })

# define the position of 4 fixed views.
tmp = 2*math.sqrt(2)
pos = ([tmp,0,tmp],[-tmp,0,tmp],[0,tmp,tmp],[0,-tmp,tmp])
dataset = 'train'
view_frames = []
for (j,p) in enumerate(pos):
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=50., sensor_width=36)
        
    frames = []

    for (i,frame) in enumerate(range(FLAGS.frame_start - 1, FLAGS.frame_end + 2)):
        
        print(frame)
      
        scene.camera.position = p
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

        matrix = scene.camera.matrix_world

        if frame >=FLAGS.frame_start and frame <= FLAGS.frame_end:
            time = (frame - FLAGS.frame_start) / (FLAGS.frame_end - FLAGS.frame_start)

            frame_data = {
              "transform_matrix": matrix.tolist(),
              "file_path": f"{dataset}/{str( (frame-FLAGS.frame_start)*4 + j).zfill(3)}",
              "time": time,
            }
            frames.append(frame_data)
    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    
    del data_stack["forward_flow"]
    del data_stack["backward_flow"]
    del data_stack["object_coordinates"]
    del data_stack["normal"]
    del data_stack["depth"]

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(data_stack["segmentation"], scene.assets, [])
    
    templates = {"rgba":"{:03d}.png"}

    # Save to image files
    kb.write_image_dict(data_stack, f'{output_dir}/dynamic_4views/view{j}',file_templates = templates)
    view_frames.append(frames)

new_frames = []
for i in range(len(frames)):
    for j in range(len(view_frames)):
        new_frames.append(view_frames[j][i])
  # --- Write the JSON descriptor for this split
kb.write_json(filename=output_dir /"dynamic_4views"/ f"transforms_{dataset}.json", data={
    "camera_angle_x": scene.camera.field_of_view,
    "frames": new_frames,
})

kb.done()



