# Modified from https://github.com/SanderGi/LCA/blob/main/scripts/lookup_fcc_photos.py

import tempfile
import requests

import pymupdf
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36"
}


def get_fcc_photos(fcc_id):
    # access the table row(s) from the FCC websites with the "internal photos" label
    response = requests.get(f"https://fcc.report/FCC-ID/{fcc_id}", headers=USER_AGENT)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # find table rows with a cell containing "Internal Photos" case insensitive
    links = []
    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td") # type: ignore
        for cell in cells:
            if "internal photos" in cell.text.lower():
                # identify the link tag in the row
                link = row.find("a") # type: ignore
                if link:
                    internal_photos_link = f'https://fcc.report{link["href"]}'
                    links.append(internal_photos_link)
                break

    # extract the images from the pdf linked on the internal photos page
    images = []
    for link in links:
        response = requests.get(f"{link}.pdf")
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(response.content)

            pdf_file = pymupdf.open(f.name)

            # iterate over PDF pages
            for page_index in range(len(pdf_file)):
                page = pdf_file.load_page(page_index)
                image_list = page.get_images()

                for img in image_list:

                    # get the XREF of the image
                    xref = img[0]

                    # extract the image bytes
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]

                    # get the image extension
                    image_ext = base_image["ext"]

                    # save the image to a file
                    with tempfile.NamedTemporaryFile(suffix=f".{image_ext}") as f:
                        f.write(image_bytes)
                        image_path = f.name
                        try:
                            images.append(plt.imread(image_path))
                        except Exception as e:
                            print(f"Error reading image: {e}")

    return images
