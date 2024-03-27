import hashlib

def create_new_file(file_name, file_content):
    with open(file_name, 'w') as f:
        f.write(file_content)

def create_file_name(pdf_name):
    hash_object = hashlib.md5(pdf_name.encode())
    return hash_object.hexdigest()
