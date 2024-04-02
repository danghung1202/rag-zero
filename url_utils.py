from urllib.parse import urlparse

def url_parser(url):
    
    parts = urlparse(url)
    segments = parts.path.strip("/").split("/")
    queries = parts.query.strip("&").split("&")
    
    elements = {
        "scheme": parts.scheme,
        "netloc": parts.netloc,
        "path": parts.path,
        "params": parts.params,
        "query": parts.query,
        "fragment": parts.fragment,
        "segments": segments,
        "queries": queries,
    }
    
    return elements

def convert_url_to_file_name(url):
    elements = url_parser(url)
    domain = elements["netloc"]
    parts = "-".join(part for part in elements["segments"])
    return domain + "-" + parts

def get_last_segment(url):
    elements = url_parser(url)
    segments = elements.get("segments")
    last_element = segments[-1]
    return last_element

def test():
    url = "https://niteco.com/about-us/"
    print(convert_url_to_file_name(url))

if __name__ == "__main__":
    test()