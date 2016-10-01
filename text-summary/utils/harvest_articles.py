# -*- coding: utf-8 -*-
"""
Downloads the articles from permissible sources.
- The Guardian

@author: Nirmalya Ghosh

Credits :
https://gist.github.com/dannguyen/c9cb220093ee4c12b840
"""

import json
import logging
import multiprocessing as mp
import os
import sys
from datetime import timedelta as td

import dateparser
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s %(processName)12s : %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _download_article_content(api_url, params, i, mssg):
    # Download the content for a specific article
    logger.info(mssg)
    content = None
    try:
        data = requests.get(api_url, params, timeout=180).json()
        if data and "response" in data:
            content = data["response"]["content"]
    except requests.exceptions.Timeout:
        logger.warn("Timeout occurred")
    except Exception, e:
        logger.warn("Some other exception: {}".format(e), exc_info=True)

    paragraph_separator = "  "
    text = None
    if content and content["type"] == "article":
        body_html = content["fields"]["body"]
        soup = BeautifulSoup(body_html, "lxml")
        paragraphs = [p.text for p in soup.find_all("p")]
        text = paragraph_separator.join(paragraphs)
        text = text.encode("ascii", "ignore")
        text = text.strip().replace("\n", " ").replace("\r", " ")

    return i, text


def _download_articles_content(cfg, json_file_path, txt_dir, date_str):
    if not os.path.exists(json_file_path):
        return

    with open(json_file_path) as json_file:
        data = json.load(json_file)

    # Read previously read articles' content
    cols = ["section", "url", "type", "headline", "wordcount", "content",
            "api_url"]
    tuples = []
    for item in data:
        headline = item["webTitle"].encode("ascii", "ignore")
        headline = headline.strip().replace("\n", " ").replace("\r", " ")
        tuples.append((item["sectionId"], item["webUrl"], item["type"],
                       headline, item["fields"]["wordcount"], "",
                       item["apiUrl"]))
    df = pd.DataFrame(tuples, columns=cols)
    df["content"].fillna("", inplace=True)
    df.content = df.content.astype(str)

    articles_file_path = os.path.join(txt_dir, "{}.txt".format(date_str))
    if os.path.exists(articles_file_path):
        df1 = pd.read_csv(articles_file_path, sep="\t", index_col=False,
                          encoding="utf-8-sig")
        df1["content"].fillna("", inplace=True)
        df1.content = df1.content.astype(str)
        df1.wordcount = df1.wordcount.astype("int64")
        df = pd.concat([df, df1])
        df = df.drop_duplicates(subset=["url"], keep="last")
        df.reset_index(drop=True, inplace=True)

    # Download content for each of the articles
    n = df.shape[0]
    params = {"api-key": cfg["guardian"]["apikey"], "format": "json",
              "show-fields": "body,lastModified,wordcount"}
    input_q = mp.Queue()
    output_q = mp.Queue()
    num_processes = cfg["guardian"]["num_processes"]
    count = 0
    for i, row in df.iterrows():
        c = row["content"]
        if c and len(c.strip()) > 0:
            continue

        # Delegate the task of downloading an article's content to a worker
        mssg = "Downloading content {} of {} for {}".format(i, n, date_str)
        input_q.put((_download_article_content,
                     (row["api_url"], params, i, mssg)))
        count += 1

    # Start the workers
    for i in range(num_processes):
        p = mp.Process(target=_worker, args=(input_q, output_q))
        p.name = "Process-{}".format(i + 1)
        p.start()

    # Read the content extracted by the workers
    for i in range(count):
        _tuple = output_q.get()
        i, content = _tuple[0], _tuple[1]
        if content:
            df.set_value(i, "content", content)

    # Stop the workers
    for i in range(num_processes):
        input_q.put("STOP")

    # Finally,
    df = df.drop_duplicates(subset=["url", "content"], keep="last")
    logger.info("Downloaded {} articles for {}".format(df.shape[0], date_str))
    df.to_csv(articles_file_path, encoding="utf-8-sig", index=False, sep="\t")


def _download_articles_list(cfg, params, date_str, json_file_path):
    if os.path.exists(json_file_path):
        return
    logger.info("Downloading articles for {}".format(date_str))

    api_endpoint = "http://content.guardianapis.com/search"
    sections_of_interest = set(cfg["guardian"]["sections"])

    all_results = []
    params["from-date"] = date_str
    params["to-date"] = date_str
    current_page, total_pages = 1, 1
    while current_page <= total_pages:
        logger.debug("Page {} of {} ..".format(current_page, total_pages))
        params["page"] = current_page
        data = None
        try:
            resp = requests.get(api_endpoint, params, timeout=180)
            data = resp.json()
        except requests.exceptions.Timeout:
            logger.warn("Timeout occurred")
            continue
        except Exception, e:
            logger.warn("Some other exception: {}".format(e), exc_info=True)
            continue

        if "response" not in data:
            logger.info(data)
            if "message" in data \
                    and "rate limit exceeded" in data["message"].lower():
                sys.exit(0)
            else:
                break

        results = data["response"]["results"]
        for result in results:
            if result["sectionId"] in sections_of_interest:
                all_results.append(result)

        # If there is more than one page
        current_page += 1
        total_pages = data["response"]["pages"]

    if len(all_results) == 0:
        logger.info("No results for {}".format(date_str))
        return

    logger.info("{} results for {}".format(len(all_results), date_str))
    with open(json_file_path, "w") as json_file:
        logging.info("Writing to {}".format(json_file_path))
        # re-serialize it for pretty indentation
        json_file.write(json.dumps(all_results, indent=2))


def _worker(in_q, out_q):
    for func, args in iter(in_q.get, "STOP"):
        result = func(*args)
        out_q.put(result)


def harvest_articles(yaml_config_file_path):
    with open(yaml_config_file_path, "r") as f:
        cfg = yaml.load(f)

    base_dir = cfg["guardian"]["base_dir"]
    d1 = dateparser.parse(cfg["guardian"]["start_date"])
    d2 = dateparser.parse(cfg["guardian"]["end_date"])

    params = {
        "api-key": cfg["guardian"]["apikey"],
        "from-date": "",
        "to-date": "",
        "order-by": "oldest",
        "page-size": 200,
        "show-fields": ["body", "lastModified", "wordcount"]
    }

    diff_range = range((d2 - d1).days + 1)
    json_file_paths = []
    for d in diff_range:
        dt = d1 + td(days=d)
        date_str = dt.strftime("%Y-%m-%d")
        year_str = dt.strftime("%Y")

        raw_dir = os.path.join(base_dir, "raw", year_str)
        txt_dir = os.path.join(base_dir, "txt", year_str)

        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        json_file_path = os.path.join(raw_dir, "{}.json".format(date_str))

        # Download the list of articles for a specific date to a raw JSON file
        _download_articles_list(cfg, params, date_str, json_file_path)
        json_file_paths.append(json_file_path)

        # Retrieve the content for each of the articles
        _download_articles_content(cfg, json_file_path, txt_dir, date_str)


if __name__ == "__main__":
    harvest_articles("../articles.yml")
