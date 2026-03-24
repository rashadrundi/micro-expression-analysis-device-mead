import json

old_prefix = "/home/rashadwsl/projects/mead-repo/micro-expression-analysis-device-mead/"
new_prefix = "C:/Users/user/Documents/code/ml/mead-repo/micro-expression-analysis-device-mead/"

data = json.load(open("train_metadata.json"))

for item in data:
    p = item["feature_path"]
    if p.startswith(old_prefix):
        item["feature_path"] = p.replace(old_prefix, new_prefix)

json.dump(data, open("train_metadata_fixed.json", "w"), indent=2)

print("Done! Saved as train_metadata_fixed.json")