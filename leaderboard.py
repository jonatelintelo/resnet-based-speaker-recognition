#!/usr/bin/env python3

# Submit a score to the leaderboard

import click
import requests
import json


@click.command()
@click.option("--baseurl", type=str, default="https://demo.spraaklab.nl/mlip/2023")
@click.option("--team", type=str, default="team00", help="team name")
@click.option("--password", type=str, default=None, help="password")
@click.option("--submit", default=None, help="Scores file to submit")
@click.option("--notes", default="", help="note to add to the submission")
def main(baseurl: str, team: str, password: str, submit: str, notes: str):
    session = requests.Session()
    login_data = dict(username=team, password=password)
    url = f"{baseurl}/auth/login"
    r = session.post(url, login_data)
    if not r.ok:
        raise ValueError("Could not login", r.reason, r.text)
    if submit is None:
        url = f"{baseurl}/api/leaderboard"
        r = session.get(url)
        if not r.ok:
            raise ValueError("Getting leaderboard not ok", r.reason)
        print(r.text)
    else:
        url = f"{baseurl}/api/submit"
        with open(submit, "rb") as file:
            r = session.post(url=url, files=dict(file=file), data=dict(notes=notes))
            if not r.ok:
                raise ValueError("Upload not ok", r.reason, r.text)
            print("Upload completed, result:", r.json())


if __name__ == "__main__":
    main()
