"""Rularea în cloud (GitHub Actions) a unui script de model: refă starea din pagina publicată (gh-pages),
   scrie configurarea GitHub din jetonul rulării și pornește scriptul exact ca pe calculator."""
import os, runpy, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
script = sys.argv[1]
repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"]
with open(os.path.join(HERE, "github.txt"), "w", encoding="utf-8") as fh:
    fh.write("repo=%s\ntoken=%s\n" % (repo, token))
# 1. istoricul rulărilor (data/) și marcajul ultimei publicări, din pagina publicată
pages = os.path.join(HERE, "_pagini")
if subprocess.call(["git", "clone", "-q", "--depth", "1", "-b", "gh-pages",
                    "https://x-access-token:%s@github.com/%s.git" % (token, repo), pages]) == 0:
    if os.path.isdir(os.path.join(pages, "data")) and not os.path.isdir(os.path.join(HERE, "data")):
        shutil.copytree(os.path.join(pages, "data"), os.path.join(HERE, "data"))
    subj = subprocess.check_output(["git", "-C", pages, "log", "-1", "--format=%s"], text=True).strip()
    if subj.startswith("date "):
        with open(os.path.join(HERE, "github_publicat.txt"), "w", encoding="utf-8") as fh:
            fh.write(subj[5:])
    shutil.rmtree(pages, ignore_errors=True)
# 2. scriptul, cu argumentele lui
sys.argv = [os.path.join(HERE, script)] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
