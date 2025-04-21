from flask import Flask, render_template, request, jsonify, send_file, url_for, request
from io import BytesIO
from PIL import Image, ImageDraw
from pathlib import Path
import pandas as pd
import numpy as np
import random
import os
import torch
import base64
import cv2
import pickle
import time
from matplotlib import pyplot as plt

from protopnet.datasets import training_dataloaders
from rashomon_sets.protorset_factory import ProtoRSetFactory

"""
To run this web server, please update the following five paths as follow:

IMAGE_DIR: Path -- should point to a universally readable/writeable directory for image serving
PATH_TO_TRAINED_RSET: Path -- should point to a fit, saved Proto-RSet
BASE_URL: str -- the URL at which this website can be accessed
RESULTS_DIR: str -- the directory to save user study results to
"""
IMAGE_DIR = None ###
PATH_TO_TRAINED_RSET = None ###
BASE_URL = None ###
RESULTS_DIR = None ###

def prep_rset():
    # os.environ["CUB200_DIR"] = "/usr/xtmp/lam135/datasets/CUB_200_2011_2/"
    # batch_sizes = {"train": 20, "project": 20, "val": 20}
    # classes_to_bias = 100

    # # Generate equally spaced values from 0 to 1
    # color_indices = np.linspace(0, 1, classes_to_bias)

    # # Get colors from the HSV colormap
    # colors = plt.cm.hsv(color_indices)

    # # Drop alpha
    # colors = torch.tensor(colors[:, :-1])
    # split_dataloaders = training_dataloaders(
    #     "cub200", 
    #     batch_sizes=batch_sizes,
    #     color_patch_params={
    #         "class_to_color": {i: colors[i] for i in range(classes_to_bias)},
    #         "patch_probability": 1.0,
    #         "patch_size": (1/5, 1/5),
    #     },
    # )

    # train_loader = split_dataloaders.project_loader
    # val_loader = split_dataloaders.val_loader
    # device = "cuda"

    # DEFAULT_RSET_ARGS = {
    #     "rashomon_bound_multiplier": 1.2,
    #     "num_models": 0,  # Right now, I'm randomly sampling models from the ellipsoid during fit
    #     "reg": "l2",
    #     "lam": 0.0001,
    #     "compute_hessian_batched": False,  # Not batching for memory's sake
    #     "max_iter": 5_000,  # The max number of iterations allowed when fitting the LR,
    #     "directly_compute_hessian": True,
    #     "device": torch.device("cuda"),
    #     "lr_for_opt": 1.0
    # }

    # rset_factory = ProtoRSetFactory(
    #     split_dataloaders=split_dataloaders,
    #     initial_protopnet_path=Path('/usr/xtmp/jcd97/proto-rset/wandb/live/artifacts/739s0fb8/54_last_only_0.4579.pth'),
    #     rashomon_set_args=DEFAULT_RSET_ARGS,
    #     device="cuda",
    #     run_complete_vislization_in_init=False,
    #     analysis_save_dir=IMAGE_DIR,
    #     reproject=False,
    #     additional_prototypes_to_sample=0
    # )
    
    with open(PATH_TO_TRAINED_RSET, "rb") as f:
        rset_factory = pickle.load(f)

    return rset_factory

app = Flask(__name__)

WORKING_RSET = None
CUR_IMG = "tmp.png"
TARGET_WIDTH = 400
RESULTS_SO_FAR = pd.DataFrame({})
LAUNCH_TIME = None

PROLIFIC_PID = None
STUDY_ID = None
SESSION_ID = None
GOLD_STD_REMOVALS = {
    52, 53, 64, 80, 106, 147, 167, 175, 176, 191, 214, 226, 241, 252, 290, 307, 403, 473, 510, 517, 563, 566, 588, 608, 632, 657, 663
}
VIABLE_REMOVALS = {
    # 66, 70, 134, 133, 176, 229, 237, 240, 300, 395, 439, 452, 479, 506, 516, 
    # 523, 541, 544, 559, 563, 567, 580, 585, 595, 602, 613, 623, 624, 627, 647, 
    # 652, 658, 661, 670, 675, 676, 677, 685, 690, 699, 704, 705,
    # 9,  11,  16,  21,  26,  31,  35,  51,  54,  60,  70,  76,  96,
    # 97, 104, 108, 110, 125, 126, 130, 135, 136, 143, 155, 160, 172,
    # 176, 177, 178, 195, 204, 206, 212, 229, 237, 240, 249, 255, 258,
    # 259, 262, 272, 279, 280, 286, 293, 300, 304, 305, 312, 318, 327,
    # 344, 359, 395, 452, 479, 488, 494, 500, 502, 505, 506, 510, 514,
    # 516, 523, 531, 536, 539, 541, 544, 551, 552, 555, 559, 563, 567,
    # 570, 580, 585, 595, 602, 613, 623, 624, 627, 640, 647, 652, 654,
    # 658, 661, 670, 675, 676, 677, 682, 685, 690, 694, 699, 704, 705
}

NO_PID_MSG = """
<h3> ERROR </h3> 
<p> 
We did not record your prolific ID, and will be unable to reimburse you. 
Please exit the study, and relaunch this study through the link on Prolific. 
If this error persists, please contact jon.donnelly@duke.edu.
</p>
"""

@app.route('/')
def informed_consent():
    global PROLIFIC_PID
    global STUDY_ID
    global SESSION_ID
    global LAUNCH_TIME
    global RESULTS_SO_FAR
    global WORKING_RSET

    PROLIFIC_PID = request.args.get('PROLIFIC_PID')
    STUDY_ID = request.args.get('STUDY_ID')
    SESSION_ID = request.args.get('SESSION_ID')
    LAUNCH_TIME = time.time()
    WORKING_RSET = prep_rset()

    try:
        RESULTS_SO_FAR = pd.read_csv(RESULTS_DIR + f"{PROLIFIC_PID}_results.csv")
    except:
        RESULTS_SO_FAR = pd.DataFrame({
            "launch_time": [],
            "time_to_action": [],
            "runtime": [],
            "action": [],
            "target": [],
            "prolific_pid": [],
            "study_id": [],
            "session_id": [],
            "success": [],
        })

    # Instructions page with a button to continue
    return render_template('informed_consent.html', base_url=BASE_URL)

@app.route('/instructions')
def instructions():
    global PROLIFIC_PID
    global LAUNCH_TIME
    if PROLIFIC_PID is None or LAUNCH_TIME is None:
        return NO_PID_MSG
    # Instructions page with a button to continue
    return render_template('instructions.html', base_url=BASE_URL)

@app.route('/main')
def main_interface():
    global PROLIFIC_PID
    global LAUNCH_TIME
    if PROLIFIC_PID is None or LAUNCH_TIME is None:
        return NO_PID_MSG
    # Main interface page
    return render_template('main_interface.html', base_url=BASE_URL, cur_img=BASE_URL + CUR_IMG)

@app.route('/stop_early', methods=['POST'])
def stop_early():
    results_for_cur_session = RESULTS_SO_FAR[
        (RESULTS_SO_FAR["prolific_pid"] == PROLIFIC_PID)
    ]
    removed_all = True
    for r in GOLD_STD_REMOVALS:
        if r not in results_for_cur_session["target"].values:
            removed_all = False
    if removed_all:
        print("TASK COMPLETED")
        return jsonify(success_message=f"Congratulations! You have removed all target prototypes in {round((start - LAUNCH_TIME) / 60, 2)} minutes. You will now be redirected to Prolific to certify completion.")
    else:
        return jsonify(message=f"WARNING: You are exiting the study early, and will not be eligible for the bonus payment. To cancel, click OK and close this alert. To confirm, enter this confirmation code on Prolific and exit the website:\n C12TG5WF")


@app.route('/process_integer', methods=['POST'])
def process_integer():
    global PROLIFIC_PID
    global LAUNCH_TIME
    if PROLIFIC_PID is None or LAUNCH_TIME is None:
        return NO_PID_MSG
    # Receive and print the integer submitted by the user
    global RESULTS_SO_FAR

    try:
        user_integers = request.json.get('integer')
        print(user_integers)
        possible_ints = [i for i in user_integers.strip().split(',')]
        realized_ints = []
        problem_inputs = []
        for j in possible_ints:
            try:
                realized_ints.append(int(j))
            except:
                problem_inputs.append(j)

        user_integers = realized_ints
    except BaseException as e:
        print(e)
        return jsonify(message=f"Error: We were unable to parse the input value {user_integers}. Full error: {e}")

    all_success = True
    for user_integer in user_integers:
        print(f"Removing {user_integer}")
        start = time.time()
        try:
            success = WORKING_RSET.require_to_avoid_prototype(user_integer)
        except BaseException as e:
            return jsonify(message=f"Error: {e}")

        all_success = all_success and success
        runtime = time.time() - start

        RESULTS_SO_FAR = pd.concat([
            RESULTS_SO_FAR,
            pd.DataFrame({
                "launch_time": [LAUNCH_TIME],
                "time_to_action": [start - LAUNCH_TIME],
                "runtime": [runtime],
                "action": ["remove_prototype"],
                "target": [user_integer],
                "prolific_pid": [PROLIFIC_PID],
                "study_id": [STUDY_ID],
                "session_id": [SESSION_ID],
                "success": [success],
            })
        ], axis=0)
    RESULTS_SO_FAR.to_csv(RESULTS_DIR + f"{PROLIFIC_PID}_results.csv")
    results_for_cur_session = RESULTS_SO_FAR[
        (RESULTS_SO_FAR["prolific_pid"] == PROLIFIC_PID)
    ]
    
    removed_all = True
    for r in GOLD_STD_REMOVALS:
        if r not in results_for_cur_session["target"].values:
            removed_all = False
    if removed_all:
        print("TASK COMPLETED")
        return jsonify(success_message=f"Congratulations! You have removed all target prototypes in {round((start - LAUNCH_TIME) / 60, 2)} minutes.  You will now be redirected to Prolific to certify completion.")

    # local_analysis_path = WORKING_RSET.display_local_analysis(user_integer)
    # print(f"FOUND PATH: {local_analysis_path}")
    # im = cv2.imread(local_analysis_path)
    # is_success, buffer = cv2.imencode(".png", im)
    # encoded_image = base64.b64encode(buffer).decode('ascii')
    # image_src = f"data:image/png;base64,{encoded_image}"
    data = request.json
    param1 = data.get('param1')
    param2 = data.get('param2')

    image_type = data.get('image_type')
    if all_success and image_type == 'Global Analysis':
        return jsonify(message=f"Prototype(s) {user_integers} successfully removed.")
    elif all_success and param1:
        return generate_image() #jsonify(message=f"Successfully removed prototype {user_integer}")
    elif all_success:
        return jsonify(message=f"Prototype(s) {user_integers} successfully removed.")
    else:
        return jsonify(message=f"Prototype(s) {user_integers} cannot be removed while maintaining accuracy.")
    # return send_file(img_io, mimetype='image/png')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    global PROLIFIC_PID
    global LAUNCH_TIME
    if PROLIFIC_PID is None or LAUNCH_TIME is None:
        return NO_PID_MSG
    # Generate an image based on dropdown selection and parameters
    data = request.json
    image_type = data.get('image_type')
    param1 = data.get('param1')
    param2 = data.get('param2')
    print(image_type, param1, param2)

    # Check validity of parameters
    try:
        if image_type == 'Prototype Grid' and int(param1) >= int(param2):
            return jsonify(message=f"Error: Max index should be greater than min index, but we recieved max={param2}, min={param1}.")
    except:
        return jsonify(message=f"Error: Please specify both the min and max prototype indices.")


    # Simple image generation based on the selection
    if image_type == 'Global Analysis':
        try:
            local_analysis_path = WORKING_RSET.display_global_analysis_for_proto(int(param1))[0]
            image = cv2.imread(str(local_analysis_path))
        except BaseException as e:
            print(e)
            return jsonify(
                message=f"Error: Prototype index {param1} is out of bounds -- there is no prototype {param1}.\n Full error: {e}"
            )
    elif image_type == 'Local Analysis':
        try:
            local_analysis_path = WORKING_RSET.display_local_analysis(int(param1))[0]
            image = cv2.imread(str(local_analysis_path))
        except BaseException as e:
            print(e)
            return jsonify(
                message=f"Error: Sample index {param1} is out of bounds -- there is no sample {param1}.\n Full error: {e}"
            )
    elif image_type == 'Prototype Grid':
        try:
            local_analysis_path = WORKING_RSET.display_proto_collage(int(param1), int(param2))
            image = cv2.imread(str(local_analysis_path))
        except IndexError as e:
            print(e)
            return jsonify(
                message=f"Error: One of prototype index {param1}, {param2} is out of bounds -- the maximum prototype index is {WORKING_RSET.initial_protopnet.prototype_layer.num_prototypes}"
            )
        except BaseException as e:
            print(e)
            return jsonify(
                message=f"Error: {e}"
            )


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PIL Image
    pil_image = Image.fromarray(image_rgb)

    # aspect_ratio = pil_image.height / pil_image.width
    # target_height = int(TARGET_WIDTH * aspect_ratio)
    # pil_image = pil_image.resize((TARGET_WIDTH, target_height))

    img_io = BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    assert False, """
    To run this web server, please update the following four paths as follows:

    IMAGE_DIR: Path -- should point to a universally readable/writeable directory for image serving
    PATH_TO_TRAINED_RSET: Path -- should point to a fit, saved Proto-RSet
    BASE_URL: str -- the URL at which this website can be accessed
    RESULTS_DIR: str -- the directory to save user study results to
    """
    app.run(host='0.0.0.0', port=8080, debug=False)
