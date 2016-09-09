# -*- coding: utf-8 -*-
"""
Downloads the photos from permissible sources, discards irrelevant photos.

@author: Nirmalya Ghosh
"""

import hashlib
import ntpath
import os
import random
import time
import urllib

import cfscrape
import cv2
import imagehash
import pandas as pd
import yaml
from PIL import Image
from bs4 import BeautifulSoup


def _calculate_image_file_name(image_url):
    return "{}.{}".format(hashlib.md5(image_url).hexdigest(),
                          image_url.split(".")[-1])


def _calculate_phash(target_dir):
    print("Calculating phash for files under {}".format(target_dir))
    files_phash = []
    for f in [f for f in os.listdir(target_dir) if
              os.path.isfile(os.path.join(target_dir, f))]:
        f = os.path.join(target_dir, f)
        try:
            files_phash.append((f, unicode(imagehash.phash(Image.open(f)))))
        except:
            pass

    return files_phash


def _delete_duplicates(base_dir, dir1, dir_final):
    # Get rid of duplicate files from raw0
    work_dir = os.path.join(base_dir, dir1)
    file_paths = [f for f in os.listdir(work_dir)]
    print(
        "{} files under {}".format(len(file_paths), work_dir))
    duplicates = set()
    for file_name in file_paths:
        if os.path.exists(os.path.join(dir_final, file_name)):
            duplicates.add(file_name)
    print("{} duplicates files under {}".format(len(duplicates), work_dir))
    ctr = 0
    for file_name in duplicates:
        os.remove(os.path.join(base_dir, dir1, file_name))
        ctr += 1
    if ctr > 0:
        print("Deleted {} duplicates files under {}".format(ctr, work_dir))


def _detect_faces_in_photo(file_path, faceCascade):
    num_faces_detected = 0
    try:
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        num_faces_detected = len(faces)
    except Exception as exc:
        print(str(exc))
    return num_faces_detected


def _delete_photos_with_faces(target_dir, cascade_xml_file_path):
    # Detect faces in the photos and delete them
    # Credit : https://realpython.com/blog/python/face-recognition-with-python/
    print("Identifying photos containing 2 or more faces..")
    photos_with_faces = {}
    faceCascade = cv2.CascadeClassifier(cascade_xml_file_path)
    for f in [f for f in os.listdir(target_dir) if
              os.path.isfile(os.path.join(target_dir, f))]:
        f = os.path.join(target_dir, f)
        num_faces_detected = _detect_faces_in_photo(f, faceCascade)
        if num_faces_detected > 1:
            print "\tFound {} face(s) in {}".format(num_faces_detected, f)
            photos_with_faces[f] = num_faces_detected
            # if len(photos_with_faces) >= 20:
            #     break

    print("{} photos in {} contain 2 or more faces. Deleting them".format(
        len(photos_with_faces), target_dir))
    for file_path, count in photos_with_faces.iteritems():
        print("\t{} contains {} faces - deleting it".format(file_path, count))
        os.remove(file_path)

    return photos_with_faces


def _delete_small_files(target_dir, minimum_size_kb):
    files_deleted = []
    for f in [f for f in os.listdir(target_dir) if
              os.path.isfile(os.path.join(target_dir, f))]:
        f = os.path.join(target_dir, f)
        if os.path.getsize(f) < minimum_size_kb * 1024:
            os.remove(f)
            files_deleted.append(f)

    print("Deleted {} files having size < {}KB" \
          .format(len(files_deleted), minimum_size_kb))
    return files_deleted


def _download_photo(photo_url, target_dirs, file_name=None):
    if file_name is None:
        file_name = _calculate_image_file_name(photo_url)
    if target_dirs is None:
        target_dirs = [os.getcwd()]

    n = len(target_dirs)
    found = [False] * n
    file_path = None
    for i, target_dir in enumerate(target_dirs):
        file_path = os.path.join(target_dir, file_name)
        found[i] = os.path.exists(file_path)
        if found[i] == True:
            break
        if i == n - 1:
            file_path = os.path.join(target_dirs[0], file_name)
            if os.path.exists(file_path) == False:
                urllib.urlretrieve(photo_url, file_path)

    return file_path


def _download_photos(models_of_interest, target_dirs, data):
    results = []
    for model in models_of_interest:
        image_urls = data[(data["is_{}".format(model)] == 1)] \
            .ImageURL.values.tolist()
        # Get the PageURL for each selected ImageURL
        df_tmp2 = data[data["ImageURL"].isin(image_urls)]
        # Iterate through the image URLs and download them
        for image_url in image_urls:
            page_url = df_tmp2[(df_tmp2.ImageURL == image_url)].PageURL.iloc[0]
            tags = df_tmp2[(df_tmp2.ImageURL == image_url)].tags.iloc[0]
            if model[0].isdigit():
                model = "B" + model
            try:
                image_file_path = _download_photo(image_url, target_dirs)
                print len(results) + 1, model.upper(), image_url
                results.append((page_url, image_url, model.upper(),
                                image_file_path, tags))
            except:
                continue

    return results


def _download_photos_listed_in_spreadsheet(base_dir, target_dirs, spreadsheet):
    file_path = os.path.join(base_dir, spreadsheet)
    df = pd.read_excel(file_path)
    df = df[(df.Size.isnull())]  # Known not to be of size 1024 x 683 (h)
    df = df.drop_duplicates(subset=["ImageURL"], keep="last")
    # Download the photos
    print("Downloading photos listed in {}".format(file_path))
    for index, row in df.iterrows():
        photo_url = row["ImageURL"]
        _ = _download_photo(photo_url, target_dirs=target_dirs)


def _harvest_photos_from_source_3(cfg, target_dirs):
    base_url = cfg["source3"]["base_url"]
    photo_id_list_file_path = cfg["source3"]["photo_id_list"]
    output_file_path = cfg["source3"]["output_file_path"]

    with open(photo_id_list_file_path) as f:
        photo_id_list = f.readlines()
    photo_id_list = [photo_id.rstrip() for photo_id in photo_id_list]
    photo_id_list = list(set(photo_id_list))
    num_photos = len(photo_id_list)

    if num_photos == 0:
        print("Nothing to be downloaded from {}".format(base_url))
        return
    else:
        print("Downloading {} photos from {}".format(num_photos, base_url))

    list_of_tuples = []
    for i, photo_id in enumerate(photo_id_list):
        page_url = base_url + photo_id
        photo_info = _scrape_photo_info_from_source_3(page_url)
        photo_url = photo_info[2]
        photo_file_path = _download_photo(photo_url, target_dirs)

        # Get the size of the photo
        img = cv2.imread(photo_file_path, cv2.CV_LOAD_IMAGE_COLOR)
        img_height, img_width, img_depth = img.shape
        img_size = "{} x {}".format(img_width, img_height)
        photo_info = list(photo_info)
        photo_info[4] = img_size
        photo_info = tuple(photo_info)
        print i, photo_info, "\n\t", photo_file_path

        list_of_tuples.append(photo_info)
        time.sleep(random.randint(5, 10))

    num_photos = len(list_of_tuples)
    if num_photos == 0:
        print("Nothing downloaded from {}".format(base_url))
        return
    else:
        print("Downloaded {} photos from {}".format(num_photos, base_url))

    # Write to a file to keep track of what has been scraped
    columns = ["Airline", "PageURL", "ImageURL", "AircraftModel", "Size",
               "Permission", "Photographer"]
    df = pd.DataFrame(list_of_tuples, columns=columns)
    if os.path.exists(output_file_path):
        df0 = pd.read_csv(output_file_path, sep="\t", encoding="utf-8-sig",
                          index_col=False)
        df = pd.concat([df, df0])
        df = df.drop_duplicates(subset=["ImageURL"], keep="last")

    df.to_csv(output_file_path, encoding="utf-8-sig", index=False, sep="\t")


def _read_yfcc_csv(file_path, models_of_interest):
    df = pd.read_csv(file_path, header=None, sep="\t")

    # The Raw file does not have a header, so assigning names to columns
    cols = df.columns.values.tolist()
    cols = ["col" + str(x) for x in cols]
    cols[6] = "desc"
    cols[8] = "tags"
    cols[10] = "latitude"
    cols[11] = "longitude"
    cols[13] = "PageURL"
    cols[14] = "ImageURL"
    cols[15] = "Attribution"
    df.columns = cols

    # Columns of interest
    cols = ["desc", "tags", "latitude", "longitude", "PageURL", "ImageURL",
            "Attribution"]
    df = df[cols]

    # Fix the contents of columns : desc, tags, ImageURL
    df["desc"] = df["desc"].str.replace("+", " ")
    df["tags"] = df["tags"].str.replace("+", " ")
    df["ImageURL_b"] = df["ImageURL"].str.replace(".jpg", "_b.jpg")
    # Rename the columns
    df.rename(columns={"ImageURL": "default_image_url"}, inplace=True)
    df.rename(columns={"ImageURL_b": "ImageURL"}, inplace=True)

    # Add the 'image_file_name' column
    def image_file_name_func(x):
        image_url = x.ImageURL
        return _calculate_image_file_name(image_url)

    df["image_file_name"] = df.apply(image_file_name_func, axis=1)

    # Parse the tags column and add a few columns
    # - This helps filter out photos wwe do not need
    df["tags"].fillna("", inplace=True)
    df["num_tags"] = 0  # Use to filter out photos with too many tags

    # Next, parse the tags to try identify the aircraft model
    # NOTE : Photos often assigned the wrong tag
    for model in models_of_interest:
        df["is_{}".format(model)] = 0
    for i, row in df.iterrows():
        tags = row["tags"].split(",")
        num_tags = len(tags)
        df.set_value(i, "num_tags", num_tags)
        if num_tags <= 15:
            # I think 15 is the maximum number of tags I'd use for a photo
            for model in models_of_interest:
                # print model, tags, model in tags
                if model in tags:
                    df.set_value(i, "is_{}".format(model), 1)

    # Filter on license
    df_tmp = df[
        (df.Attribution == "Attribution-NonCommercial-NoDerivs License")]

    # Finally,
    df = df_tmp
    return df


def _resize_photo(file_path, target_width=1024, target_height=683):
    # Helps conform to a standard size
    # Credit :
    # https://enumap.wordpress.com/2014/07/06/python-opencv-resize-image-by-width/
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
    # http://stackoverflow.com/a/22907694
    original = cv2.imread(file_path, cv2.CV_LOAD_IMAGE_COLOR)
    if original is None:
        # Corrupt file
        return False
    original_height, original_width, original_depth = original.shape
    scale_w = float(target_width) / original_width
    scale_h = float(target_height) / original_height

    if scale_w == 1.0:
        newX, newY = original.shape[1], original.shape[0]
        if scale_h > 1.0:
            newX, newY = original.shape[1], original.shape[0]
        elif scale_h == 1.0:
            return False
        if original_height - target_height < 10:
            newX, newY = original.shape[1], target_height
    else:
        newX, newY = original.shape[1] * scale_w, original.shape[0] * scale_w

    modified = cv2.resize(original, (int(newX), int(newY)))

    # Resizing a photo might require adding a border to the top
    # if the height of the rescaled image is less than the desired height
    border = [0] * 4  # top, bottom, left, right
    if modified.shape[0] < target_height:
        border[0] = target_height - modified.shape[0]
    elif modified.shape[0] > target_height:
        # Need to crop
        cropped = modified[0:target_height, 0:target_width]
        dims = cropped.shape
        if dims[0] == target_height and dims[1] == target_width:
            modified = cropped

    # Add a border to the top (if required)
    cv2.copyMakeBorder(modified, border[0], border[1], border[2], border[3],
                       cv2.BORDER_REPLICATE)

    # Rewrite to file
    cv2.imwrite(file_path, modified)
    return True


def _resize_photos(target_dir, target_width=1024, target_height=683):
    photos_resized = []
    print("Resizing photos under {} to {} x {}" \
          .format(target_dir, target_width, target_height))
    for f in [f for f in os.listdir(target_dir) if
              os.path.isfile(os.path.join(target_dir, f))]:
        f = os.path.join(target_dir, f)
        is_resized = _resize_photo(f, target_width, target_height)
        if is_resized:
            photos_resized.append(f)

    print("Resized {} photos under {}".format(len(photos_resized), target_dir))
    return photos_resized


def _scrape_photo_info_from_source_3(page_url):
    scraper = cfscrape.create_scraper()
    scraped_content = scraper.get(page_url).content
    soup = BeautifulSoup(scraped_content, "lxml")
    photos = soup.find_all("img", class_="main-image")
    photo_url = photos[0]["src"]

    # Scrape the aircraft model and airline
    aircraft_model, airline = None, None
    info_section = soup.find("section", class_="additional-info aircraft")
    p_elems = info_section.select("p")
    for p_elem in p_elems:
        text = p_elem.text.strip()
        if len(text) > 0:
            if "Aircraft: " in text:
                aircraft_model = text.split(":")[1].strip()
            if "Airline: " in text:
                airline = text.split(":")[1].strip()

    # Scrape the photographer's name
    photographer_name = None
    info_section = soup.find("section", class_="additional-info photographer")
    p_elems = info_section.select("p")
    for i, p_elem in enumerate(p_elems):
        text = p_elem.text.strip()
        if len(text) > 0:
            if i == 0:
                photographer_name = text.strip()
    size = ""  # Placeholder - we set it after we download the photo
    return ((airline, page_url, photo_url, aircraft_model, size, "No",
             photographer_name))


def harvest_photos(cfg):
    models_of_interest = ["a320", "a321", "a330", "a340", "a350", "a380",
                          "737", "747", "757", "767", "777", "787"]
    df = _read_yfcc_csv(cfg["data"]["yfcc_csv"], models_of_interest)

    base_dir = cfg["data"]["download_base_dir"]
    raw0, raw1 = "raw0", "raw1"

    # Downloading the photos
    final_dir = os.path.join(base_dir, "selected")
    target_dirs = [os.path.join(base_dir, raw0),
                   os.path.join(base_dir, raw1),
                   os.path.join(base_dir, final_dir)]

    # Adding aircraft-model specific subdirectories (if any)
    for m in cfg["data"]["aircraft_model_subdirectory_names"]:
        directory = os.path.join(final_dir, m)
        if not os.path.exists(directory):
            os.makedirs(directory)

    final_dir_subdirs = [x[0] for x in os.walk(final_dir)]
    if final_dir_subdirs and len(final_dir_subdirs) > 1:
        final_dir_subdirs.remove(final_dir)
        target_dirs.extend(final_dir_subdirs)

    download = cfg["data"]["download_yhcc"]
    results_file_path = "harvested_images_with_filepaths.txt"  # TODO get rid
    if download == True:
        results = _download_photos(models_of_interest, target_dirs, df)
        cols = ["PageURL", "ImageURL", "AircraftModel", "ImageFilePath",
                "Tags"]
        results_df = pd.DataFrame(results, columns=cols)
        # Keep record of what we have harvested so far from Flickr
        results_df.to_csv(results_file_path, index=False, sep="\t")
    else:
        results_df = pd.read_csv(results_file_path, index_col=False, sep="\t")

    # Downloading the images, from source 3
    _harvest_photos_from_source_3(cfg, target_dirs)

    # Read the file listing the irrelevant photos - this list is hand curated
    irlvnt = pd.read_csv(cfg["data"]["irrelevant_photos"], sep="\t")
    irlvnt["is_irrelevant_image"] = 1

    # Get rid of the irrelevant images
    print("Getting rid of the {} irrelevant images".format(irlvnt.shape[0]))
    df = pd.merge(df, irlvnt, how="left")
    df["is_irrelevant_image"].fillna(0, inplace=True)
    df = df[(df.is_irrelevant_image == 0)]
    # Also, get rid of them from the folder, if exists
    for file_name in irlvnt.image_file_name.values.tolist():
        file_path = os.path.join(base_dir, raw0, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print("\tDeleted {}".format(file_path))

    # Next,
    _download_photos_listed_in_spreadsheet(base_dir, target_dirs,
                                           cfg["data"]["spreadsheet"])

    # Delete photos which are too small (less than 10KB)
    _delete_small_files(target_dirs[0], cfg["data"]["min_size_kb"])
    # TODO remove files deleted from the results_df

    # Resize the photos
    _resize_photos(target_dirs[0], target_width=1024, target_height=683)

    # The YFCC dataset has many irrelevant / incorrectly tagged photos,
    # one way to automatically reduce the number of irrelevant photos
    # is to discard those with 2 or more faces in it
    cascade_xml_file_path = cfg["data"]["haar_cascade_xml"]
    photos_with_faces = _delete_photos_with_faces(target_dirs[0],
                                                  cascade_xml_file_path)
    if photos_with_faces:
        # Need to update irrelevant_photos
        file_names = []
        for file_path, count in photos_with_faces.iteritems():
            file_names.append(ntpath.basename(file_path))
        df_tmp = pd.DataFrame()
        df_tmp["image_file_name"] = file_names
        df_tmp["is_irrelevant_image"] = 1
        irlvnt = pd.concat([irlvnt, df_tmp])
        irlvnt = irlvnt.drop_duplicates(subset=["image_file_name"],
                                        keep="last")
        irlvnt.to_csv(cfg["data"]["irrelevant_photos"], index=False, sep="\t")

    # Get rid of duplicate files from raw0
    _delete_duplicates(base_dir, raw0, final_dir)
