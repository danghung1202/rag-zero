from urllib.parse import urlparse

def url_parser(url):
    
    parts = urlparse(url)
    directories = parts.path.strip("/").split("/")
    queries = parts.query.strip("&").split("&")
    
    elements = {
        "scheme": parts.scheme,
        "netloc": parts.netloc,
        "path": parts.path,
        "params": parts.params,
        "query": parts.query,
        "fragment": parts.fragment,
        "directories": directories,
        "queries": queries,
    }
    
    return elements

def convert_url_to_file_name(url):
    elements = url_parser(url)
    print(elements)
    domain = elements["netloc"]
    parts = "-".join(part for part in elements["directories"])
    return domain + "-" + parts

def test():
    url = "https://niteco.com/about-us/"
    print(convert_url_to_file_name(url))

if __name__ == "__main__":
    test()