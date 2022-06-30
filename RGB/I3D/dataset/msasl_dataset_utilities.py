import pathlib


def msasl_num_class(root_dir: str) -> int:
    path = pathlib.Path(root_dir)

    classes = []
    for file in path.iterdir():
        class_num = file.stem[0:3]
        if class_num not in classes:
            classes.append(class_num)

    return len(classes)


def msasl_make_dataset(split_file: str, split: str, root_dir: str, mode: str,
                       num_classes: int) -> list:
    return
