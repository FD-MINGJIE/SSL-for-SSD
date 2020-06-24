class MixUp(object):
    def __init__(self, alpha1=1.5, alpha2=1.5):
        self.a1 = alpha1 
        self.a2 = alpha2
        self.lambd = -100
    def forward(self, img, boxes=None, labels=None):
        lambd = 1
        lambd = max(0, min(1, np.random.beta(self.a1, self.a2)))
        if lambd >= 1:
            return img, boxes, labels
        img_2 = copy.deepcopy(img[indices])
        for i in range(img.shape[0]):
            img1 = img[i]
            img2 = img_2[i]
            height = max(img1.shape[0], img2.shape[0])
            width = max(img1.shape[1], img2.shape[1])
            mix_img = np.zeros((height, width, 3))
            mix_img[:img1.shape[0], :img1.shape[1], :] = img1 * lambd
            mix_img[:img2.shape[0], :img2.shape[1], :] += img2 * (1. - lambd) 
            mix_img = mix_img.astype('uint8')
 
        self.lambd = lambd    
        return img, boxes, labels
