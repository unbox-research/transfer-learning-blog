# -*- coding: utf-8 -*-
""" dl.py

	A module to search google and download images corresponding 
	to search terms. From:

		https://github.com/hardikvasa/google-images-download
"""

# _______________________________________________________________________
# Imports

from google_images_download import google_images_download

# _______________________________________________________________________
# Download images by search terms.

response = google_images_download.googleimagesdownload()
# Choose what to search for here. 
args = ['cat', 'dog']
		
# If not extant, a `downloads` directory is created and populated
# with subdirs (folders) with the same name as the search terms.

# (This is ideal for our use case, where we will retrain an
# image classifier and want data to be in just this format.)

def run():
	for arg in args:
		absolute_image_paths = response.download({'keywords' : arg, 
												  'limit': 10,  # Requires `chromedriver` for more than 100 image scrapes.
												  # To download: https://sites.google.com/a/chromium.org/chromedriver/downloads (link live 8/30/18)
												   'chromedriver': '/Users/wnowak/work/unbox/openhands/img_scraping/chromedriver'})

if __name__ == '__main__':
	run()