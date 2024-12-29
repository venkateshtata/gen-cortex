import subprocess


def fetch_data(URL, OUTPUT_DIR):
    fetch_command = [
        "wget",
        "-e", "robots=off",
        "--recursive",
        "--no-clobber",
        "--page-requisites",
        "--html-extension",
        "--convert-links",
        "--restrict-file-names=windows",
        "--domains", "docs.ray.io",
        "--no-parent",
        "--accept=html",
        "-P", OUTPUT_DIR,
        URL,
    ]
    
    subprocess.run(fetch_command, check=True)
