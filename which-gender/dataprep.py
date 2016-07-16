# -*- coding: utf-8 -*-
"""
Prepares the daataset.

This script reads a ZIP file containing 681288 blog posts (downloaded from 
http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm). The downloaded ZIP file contains
19000+ XML files, each containing posts by one author. This script reads the
XML files and creates a dataset to speed up work of the other notebooks.

@author: Nirmalya Ghosh
"""

import Queue
import datetime
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

import dateparser
import pandas as pd
import yaml
from bs4 import BeautifulSoup


def read_xml_str(xml_str):
    blog = BeautifulSoup(xml_str, "lxml")
    dates, posts = [], []
    if blog is not None:
        for d in blog.findAll("date"):
            dates.append(d.text)
        for d in blog.findAll("post"):
            posts.append(d.text.replace("\r\n\t", "").strip())

    return dates, posts


def read_zip_file(zip_file_path):
    zf = zipfile.ZipFile(zip_file_path, "r")
    columns = ["blogger_id", "gender", "age", "industry", "astro_sign"]
    metadata = []
    blog_posts = []
    for info in zf.infolist():
        if len(info.filename) < 7:
            continue

        blog_key = info.filename[6:]
        attrbs = blog_key.split(".")[:-1]
        metadata.append(attrbs)

        if info.filename[-4:] == ".xml":
            xml_str = zf.read(info.filename)
            blog_posts.append((attrbs[0], xml_str))

    metadata_df = pd.DataFrame(metadata, columns=columns)
    return metadata_df, blog_posts


def worker(q_in, q_out):
    while True:
        if q_in.empty() == True:
            break
        _tuple_ = q_in.get()
        blogger_id = _tuple_[0]
        dates, posts = read_xml_str(_tuple_[1])
        for i in range(min(len(dates), len(posts))):
            q_out.put((blogger_id, dates[i], posts[i]))
        q_in.task_done()


def writer(q_out, timeout_seconds, output_file_path):
    fmt2 = "%Y-%m-%d"
    f = open(output_file_path, "w")
    f.write("blogger_id\tdate\tblog_post\n")
    l = []
    while True:
        try:
            item = q_out.get(True, timeout_seconds)
            l.append(item)
        except Queue.Empty:
            print "Writer has finished its job"
            break  # Work done
        if len(l) > 150:
            for t in l:
                dt_str = format_post_date(t[1], fmt2)
                f.write("{}\t{}\t{}\n"
                        .format(t[0], dt_str, format_post_text(t[2])))
            l = []
        q_out.task_done()
    
    # If any left,
    if len(l) > 0:
        for t in l:
            dt_str = format_post_date(t[1], fmt2)
            f.write("{}\t{}\t{}\n"
                    .format(t[0], dt_str, format_post_text(t[2])))
    
    f.close()


def format_post_date(date_str, date_fmt_str):
    # Dates specified in English, French, etc.
    dt = dateparser.parse(date_str) if date_str and len(date_str) > 0 else None
    return dt.strftime(date_fmt_str) if dt else ""


def format_post_text(text):
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ") \
        .encode("utf-8")


if __name__ == '__main__':
    t0 = time.time()
    zip_file_path = os.path.join("datasets", "blogs.zip")
    metadata_df, blog_posts = read_zip_file(zip_file_path)
    metadata_df.to_csv(os.path.join("datasets", "blog_posts_metadata.txt"),
                       sep="\t", index=False)

    input_q = mp.JoinableQueue()
    output_q = mp.JoinableQueue()

    for _tuple_ in blog_posts:
        input_q.put(_tuple_)

    # Setup (then start) list of processes responsible for parsing the XML
    processes = []
    for i in range(5):
        p = mp.Process(target=worker, args=(input_q, output_q))
        p.daemon = True
        processes.append(p)
    for p in processes:
        p.start()
    print "Started {} processes for parsing the XML".format(len(processes))

    # Start the writer process
    output_file_path = os.path.join("datasets", "blog_posts.txt")
    writer_p = mp.Process(target=writer,
                          args=(output_q, 15, output_file_path,))
    writer_p.daemon = True
    writer_p.start()
    print "Started the writer process"

    for p in processes:
        p.join()
    writer_p.join()
    print "Time taken : {:.2f} seconds".format(time.time() - t0)
