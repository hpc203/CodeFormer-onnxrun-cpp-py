import argparse
import cv2
import numpy as np
import onnxruntime as ort


class CodeFormer():
    def __init__(self, modelpath):
        # net = cv2.dnn.readNet(modelpath)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(modelpath, so)
        model_inputs = self.session.get_inputs()
        self.input_name0 = model_inputs[0].name
        self.input_name1 = model_inputs[1].name
        self.inpheight = model_inputs[0].shape[2]
        self.inpwidth = model_inputs[0].shape[3]

    def post_processing(self, tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
        # tensor 3ch
        _tensor = tensor[0]

        _tensor = _tensor.clip(min_max[0], min_max[1])

        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.ndim

        if n_dim == 3:
            img_np = _tensor
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        return img_np

    def detect(self, srcimg):
        dstimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        dstimg = cv2.resize(dstimg, (self.inpwidth, self.inpheight), interpolation=cv2.INTER_LINEAR)
        dstimg = (dstimg.astype(np.float32)/255.0 - 0.5) / 0.5
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Inference
        output = self.session.run(None, {self.input_name0: input_image, self.input_name1:np.array([0.5])})[0]
        restored_img = self.post_processing(output, rgb2bgr=True, min_max=(-1, 1))
        return restored_img.astype('uint8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='input.png', help="image path")
    parser.add_argument("--modelpath", type=str, default='codeformer.onnx', help="onnxmodel path")
    args = parser.parse_args()

    mynet = CodeFormer(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    restored_img = mynet.detect(srcimg)
    restored_img = cv2.resize(restored_img, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_LINEAR)

    # if srcimg.shape[0]>=srcimg.shape[1]:
    #     result = np.vstack((srcimg, restored_img))
    # else:
    #     result = np.hstack((srcimg, restored_img))

    # cv2.imwrite('result.jpg', restored_img)
    cv2.namedWindow("srcimg", cv2.WINDOW_NORMAL)
    cv2.imshow("srcimg", srcimg)
    cv2.namedWindow("restored_img", cv2.WINDOW_NORMAL)
    cv2.imshow("restored_img", restored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
