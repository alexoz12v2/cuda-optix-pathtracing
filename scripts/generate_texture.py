import os
import sys
import json

# case insensitive
type_mapping = {
    "basecolor": "diffuse",
    "diffuse": "diffuse",
    "normal": "normal",
    "roughness": "roughness",
    "metal": "metallic",
    "metallic": "metallic",
    "metalness": "metallic"
}

def process_texture(path):
    # extract file name without extension
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)

    if not ext:
        return

    type_key = None
    for key in type_mapping:
        if name.lower().endswith(key):
            type_key = type_mapping[key]
            break

    if not type_key:
        return

    output = {
        "name": name,
        "type": type_key,
        "path": path
    }

    return output


if __name__ == "__main__":
    # usage: "python generate_texture.py <texture_path>"
    # if len(sys.argv) != 2:
    #     print("Usage: python generate_texture.py <texture_path>")
    #     sys.exit(1)
    # texture_path = sys.argv[1]
    # output = process_texture(texture_path)
    # print(json.dumps(output, indent=2))

    # usage: "cat texture_paths.txt | python generate_texture.py" (or find or anything else), separated by newline
    paths = [line.strip() for line in sys.stdin if line.strip()]
    objects = [process_texture(p) for p in paths if process_texture(p) is not None]
    print(json.dumps(objects, indent=2))
