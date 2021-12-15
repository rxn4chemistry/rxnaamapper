"""Utily function to fetch active site from Pfam"""

import json
import re
from typing import List, Tuple

import requests
import xmltodict

HEADERS = {"Expect": ""}
FILES = {"seq": None, "output": (None, "xml")}
JOB_ID_URL = "https://pfam.xfam.org/search/sequence"


def get_result_url(aa_sequence: str) -> Tuple[str, str]:
    """Get the job id and the result url from Pfam sequence search
    Args:
        aa_sequence: amino acid sequence used to perform the sequence search

    Return:
        job_id and result_url used to get the outcomes from pfam
    """
    FILES["seq"] = (None, aa_sequence)
    response = requests.post(JOB_ID_URL, headers=HEADERS, files=FILES)
    response_dict = xmltodict.parse(response.content)
    response_dict = json.dumps(response_dict)
    response_dict = json.loads(response_dict)
    jobs = response_dict.get("jobs")
    if jobs is None:
        raise ValueError("jobs not found")
    job = jobs.get("job")
    if job is None:
        raise ValueError("job not found")
    job_id = job["@job_id"]
    result_url = job["result_url"]
    return job_id, result_url


def get_active_site(result_url: str) -> List[Tuple[int, int]]:
    """Fetch the active site information
    Args:
        job_id: job id given by the Pfam API

    Returns:
        list of the interval described the binding sites
    """
    url = f"https://pfam.xfam.org{result_url}"
    response = requests.get(url)
    response_dict = xmltodict.parse(response.content)
    response_dict = json.dumps(response_dict)
    response_dict = json.loads(response_dict)

    # parse the response dictionary
    pfam = response_dict["pfam"]
    results = pfam["results"]
    matches = results["matches"]
    protein = matches["protein"]
    database = protein["database"]

    # keep the match with the highest score according to Pfam
    match = database["match"]
    if isinstance(match, list):
        match = match[0]
    location = match["location"]
    if isinstance(location, list):
        location = location[0]
    ali_start = int(location["@ali_start"]) - 1
    pp = location["pp"]
    pp = re.sub(r"\.+", "", pp)

    active_sites = [
        (m.start() + ali_start, m.end() + ali_start) for m in re.finditer(r"\*+", pp)
    ]

    return active_sites
