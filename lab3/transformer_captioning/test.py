import requests

def check_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def filter_urls(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            url = line.strip()
            if check_url(url):
                outfile.write(url + '\n')

# Filter URLs for train and validation datasets
filter_urls('./datasets/coco_captioning/train2014_urls.txt', './datasets/coco_captioning/train2014_valid_urls.txt')
filter_urls('./datasets/coco_captioning/val2014_urls.txt', './datasets/coco_captioning/val2014_valid_urls.txt')