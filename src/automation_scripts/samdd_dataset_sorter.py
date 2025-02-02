import os
import shutil
from pathlib import Path

_BEHAVIORS = {
    "Texting Right": "5",
    "Texting Left": "4",
    "Talking using Phone Right": "3",
    "Talking using Phone Left": "2",
    "Drinking": "1",
    "Drive Safe": "0",
    "Head Down": "9",
    "Look Behind": "8"
    # Operating Radio
}

# File path of each tester
_TESTERS_PATH = "F:\\Jownjown\\Education\\4th Year\\1st Sem\\Thesis 1\\Datasets\\Singapore AutoMan Distracted Driving\\SAM-DD(RGB)\\Evaluation\\Raw"
_FOLDER_SORT_PATH = "F:\\Jownjown\\Education\\4th Year\\1st Sem\\Thesis 1\\Datasets\\Singapore AutoMan Distracted Driving\\SAM-DD(RGB)\\Evaluation"


def sort():

    count = 0

    for behavior, behavior_code in _BEHAVIORS.items():  # For each behavior
        print(behavior)
        for tester in os.listdir(_TESTERS_PATH):  # For each tester
            # Navigate to folder
            folder_behavior = os.path.join(
                _TESTERS_PATH, tester, behavior_code, "side_RGB")

            # Extract numerical part for correct sorting
            frames = sorted(
                os.listdir(folder_behavior),
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )

            for frame in frames:
                count += 1
                src_path = os.path.join(folder_behavior, frame)
                dest_path = os.path.join(
                    _FOLDER_SORT_PATH, behavior, f"{behavior.replace(' ', '_')}_{count:05d}.jpg")
                shutil.copy2(src_path, dest_path)


sort()
