import cv2
import numpy as np

from .bwm_core import WaterMarkCore


class WaterMark:
    def __init__(
        self,
        password_wm=1,
        password_img=1,
        block_shape=(4, 4),
        mode="common",
        processes=None,
    ):

        self.bwm_core = WaterMarkCore(
            password_img=password_img, mode=mode, processes=processes
        )

        self.password_wm = password_wm

        self.alpha = None  # image's third dimension

    def read_img(self, filename: str):
        img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
        if not img:
            raise IOError(f"Image file '{filename}' not read")

        self.alpha = None
        if img.shape[2] == 4 and img[:, :, 3].min() < 255:
            self.alpha = img[:, :, 3]
            img = img[:, :, :3]

        self.bwm_core.read_img_arr(img=img)
        return img

    def read_img_wm(self, filename):
        wm = cv2.imread(filename)
        if not wm:
            raise IOError(f"Image file '{filename}' not read")

        # Convert the watermark in one-dimensional bit format
        self.wm = wm[:, :, 0]
        # Encrypted information only uses the bit class, discarding the gray level
        self.wm_bit = self.wm.flatten() > 128

    def read_wm(self, wm_content, mode="img"):
        if mode == "img":
            self.read_img_wm(filename=wm_content)
        elif mode == "str":
            byte = bin(int(wm_content.encode("utf-8").hex(), base=16))[2:]
            self.wm_bit = np.array(list(byte)) == "1"
        elif mode == "array":
            self.wm_bit = np.array(wm_content)
        else:
            raise ValueError(f"The reading watermark mode '{mode} is not supported.'")

        self.wm_size = self.wm_bit.size

        # watermark encryption:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename):
        embed_img = self.bwm_core.embed()
        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])

        cv2.imwrite(filename, embed_img)
        return embed_img

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename, wm_shape, out_wm_name=None, mode="img"):
        img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)

        self.wm_size = np.array(wm_shape).prod()

        if mode in ("str", "bit"):
            wm_avg = self.bwm_core.extract_with_kmeans(img=img, wm_shape=wm_shape)
        else:
            wm_avg = self.bwm_core.extract(img=img, wm_shape=wm_shape)

        # decrypt
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # Convert to the specified format：
        if mode == "img":
            cv2.imwrite(out_wm_name, 255 * wm.reshape(wm_shape[0], wm_shape[1]))
        elif mode == "str":
            byte = "".join((np.round(wm)).astype(np.int).astype(np.str))
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode(
                "utf-8", errors="replace"
            )

        return wm
