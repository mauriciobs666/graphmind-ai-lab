"""Operator command runner (a taint chain: source, middle hop, sink)."""
import subprocess

from fastapi import Query


def run_cmd(cmd: str = Query(...)):
    return exec_cmd(normalize(cmd))


def normalize(cmd):
    return cmd.strip()


def exec_cmd(cmd):
    return subprocess.run(cmd, shell=True)
