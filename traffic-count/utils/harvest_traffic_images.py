# -*- coding: utf-8 -*-
"""
Downloads the traffic images.

@author: Nirmalya Ghosh
"""

import logging
import os
import shutil
import time
from datetime import datetime as dt
from datetime import timedelta as td

import dateparser
import requests
import yaml

logging.basicConfig(format="%(asctime)s %(processName)12s : %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

headers = {"api-key": None, "accept": "application/json"}
url = "https://api.data.gov.sg/v1/transport/traffic-images"


def download_image(image_url, image_file_path):
    response = requests.get(image_url, stream=True)
    with open(image_file_path, "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


def download_images(date_time, base_dir=None, camera_ids=None, wait_secs=15):
    date_time_str = date_time.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info("Downloading images for {}".format(date_time_str))
    payload = {"date_time": date_time_str}
    data = requests.get(url, headers=headers, params=payload).json()
    ts = data["items"][0]["timestamp"]
    cameras = data["items"][0]["cameras"]  # 68
    logger.info("# Cameras for {} : {}".format(ts, len(cameras)))
    for camera in cameras:
        img_url = camera["image"]
        camera_id = int(camera["camera_id"])
        if camera_ids and camera_id not in camera_ids:
            continue

        cam_ts_str = camera["timestamp"]
        cam_ts_str = cam_ts_str[:-6] if len(cam_ts_str) > 20 else cam_ts_str
        camera_ts = dt.strptime(cam_ts_str, "%Y-%m-%dT%H:%M:%S")
        # NOTE: camera timestamp may be different (and lagging by more
        #       than a minute) from the timestamp outside this loop
        file_name = "{}-{}-{}".format(camera_id,
                                      camera_ts.strftime("%Y%m%d-%H%M%S"),
                                      img_url.split("/")[-1].replace("-", ""))
        year = camera_ts.strftime("%Y")
        month = camera_ts.strftime("%m")
        image_dir = os.path.join(base_dir, year, month, str(camera_id))
        image_file_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(image_file_path):
            download_image(img_url, image_file_path)

        time.sleep(wait_secs)


def harvest_traffic_images(yaml_config_file_path):
    with open(yaml_config_file_path, "r") as f:
        cfg = yaml.load(f)

    global headers
    headers["api-key"] = cfg["harvest"]["apikey"]
    d1 = dateparser.parse(cfg["harvest"]["start_date_time"])
    d2 = dateparser.parse(cfg["harvest"]["end_date_time"])
    base_dir = cfg["harvest"]["base_dir"]
    if base_dir is None:
        base_dir = os.getcwd()

    camera_ids = cfg["harvest"]["camera_ids"]
    if camera_ids and len(camera_ids) > 0:
        camera_ids = set(camera_ids)
    else:
        camera_ids = None

    wait_secs = cfg["harvest"]["wait_seconds"]

    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    diff_range = range((d2 - d1).days * 24 * 60 + 1)
    for d in diff_range:
        date_time = d1 + td(minutes=d)
        download_images(date_time, base_dir, camera_ids, wait_secs)


if __name__ == "__main__":
    harvest_traffic_images("../traffic-count.yml")
