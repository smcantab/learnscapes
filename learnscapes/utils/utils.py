def select_device(dev="cpu"):
    if dev == "cpu":
        return "/cpu:0"
    else:
        return "/gpu:0"
