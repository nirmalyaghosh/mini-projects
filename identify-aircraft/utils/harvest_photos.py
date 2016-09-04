# -*- coding: utf-8 -*-
"""
Downloads the photos from permissible sources, discards irrelevant photos.

@author: Nirmalya Ghosh
"""

import hashlib
import os
import urllib

import imagehash
import pandas as pd
import yaml
from PIL import Image


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


def _download_image(image_url, target_dirs, image_file_name=None):
    if image_file_name is None:
        image_file_name = _calculate_image_file_name(image_url)
    if target_dirs is None:
        target_dirs = [os.getcwd()]

    n = len(target_dirs)
    found = [False] * n
    for i, target_dir in enumerate(target_dirs):
        image_file_path = os.path.join(target_dir, image_file_name)
        found[i] = os.path.exists(image_file_path)
        if i == n - 1 and found[i] == False:
            image_file_path = os.path.join(target_dirs[0], image_file_name)
            if os.path.exists(image_file_path) == False:
                urllib.urlretrieve(image_url, image_file_path)

    return image_file_path


def _download_images(models_of_interest, target_dirs, data):
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
                image_file_path = _download_image(image_url, target_dirs)
                print len(results) + 1, model.upper(), image_url
                results.append((page_url, image_url, model.upper(),
                                image_file_path, tags))
            except:
                continue

    return results


def _download_images_listed_in_spreadsheet(base_dir, target_dirs, spreadsheet):
    file_path = os.path.join(base_dir, spreadsheet)
    df = pd.read_excel(file_path)
    df = df[(df.Size.isnull())]  # Known not to be of size 1024 x 683 (h)
    df = df.drop_duplicates(subset=["ImageURL"], keep="last")
    # Download the images
    print("Downloading images listed in {}".format(file_path))
    for index, row in df.iterrows():
        image_url = row["ImageURL"]
        image_file_path = _download_image(image_url, target_dirs=target_dirs)


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


def harvest_photos(cfg):
    models_of_interest = ["a320", "a321", "a330", "a340", "a350", "a380",
                          "737", "747", "757", "767", "777", "787"]
    df = _read_yfcc_csv(cfg["data"]["yfcc_csv"], models_of_interest)

    # Read the file listing the irrelevant photos - this list is hand curated
    irlvnt = pd.read_csv(cfg["data"]["irrelevant_photos"])
    irlvnt["is_irrelevant_photo"] = 1

    # Get rid of the irrelevant photos
    print("Getting rid of the {} irrelevant photos".format(irlvnt.shape[0]))
    df = pd.merge(df, irlvnt, how="left")
    df["is_irrelevant_photo"].fillna(0, inplace=True)
    df = df[(df.is_irrelevant_photo == 0)]
    # Also, get rid of them from the folder, if exists
    base_dir = cfg["data"]["download_base_dir"]
    raw0, raw1 = "raw0", "raw1"
    for file_name in irlvnt.image_file_name.values.tolist():
        file_path = os.path.join(base_dir, raw0, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print("\tDeleted {}".format(file_path))

    # Downloading the images
    final_dir = os.path.join(base_dir, "selected")
    target_dirs = [os.path.join(base_dir, raw0),
                   os.path.join(base_dir, raw1),
                   final_dir]
    download = cfg["data"]["download_yhcc"]
    results_file_path = "harvested_photos_with_filepaths.txt"  # TODO get rid
    if download == True:
        results = _download_images(models_of_interest, target_dirs, df)
        cols = ["PageURL", "ImageURL", "AircraftModel", "ImageFilePath",
                "Tags"]
        results_df = pd.DataFrame(results, columns=cols)
        # Keep record of what we have harvested so far from Flickr
        results_df.to_csv(results_file_path, index=False, sep="\t")
    else:
        results_df = pd.read_csv(results_file_path, index_col=False, sep="\t")

    # Next,
    _download_images_listed_in_spreadsheet(base_dir, target_dirs,
                                           cfg["data"]["spreadsheet"])

    # Delete images which are too small (less than 10KB)
    _delete_small_files(target_dirs[0], cfg["data"]["min_size_kb"])
    # TODO remove files deleted from the results_df

    # # Get rid of duplicate files from raw0
    _delete_duplicates(base_dir, raw0, final_dir)
