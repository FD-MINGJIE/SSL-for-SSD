class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=3, length=30):   
        self.n_holes = n_holes
        self.length = length

    def forward(self, img, boxes=None, labels=None):
        # [H,W,C]
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w, 3), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        img = torch.from_numpy(img.copy()) * mask
        return img.numpy(), boxes, labels
